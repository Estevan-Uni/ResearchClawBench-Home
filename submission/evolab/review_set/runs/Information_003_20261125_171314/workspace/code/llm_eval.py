"""LLM comparison: a locally-run open instruction-tuned LLM (Qwen2.5-1.5B-Instruct, CPU) classifies
100 text-serialised test flows (10 per class) with an in-context few-shot prompt (3 labelled
training flows per class). Classes are mapped to letters A-J and the prediction is the argmax of the
next-token logits restricted to those letters. The shared prompt prefix is KV-cached once.
Reports macro-F1/binary-F1 and wall-clock latency; compares to RF / DIDS-MFL on the same flows."""
import sys, time, glob, copy, numpy as np, torch
sys.path.insert(0, '/workspace/code')
from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)  # (results json label corrected to 4 threads)
D = load(); y = D['attack']; n = len(y); tr, va, te = chrono_split(n)
Fq = np.load(f'{OUT}/Fq.npy')
emb = np.load(f'{OUT}/runs/known_dids_mfl_s0_emb.npz')
rng = np.random.default_rng(7)
qidx = np.concatenate([rng.choice(emb['gid'][emb['y'] == c], 10, replace=False) for c in range(10)])
LET = 'ABCDEFGHIJ'
from sklearn.feature_selection import mutual_info_classif
_sub = rng.choice(tr, 20000, replace=False)
FEATS = np.argsort(-mutual_info_classif(Fq[_sub], y[_sub], random_state=0))[:20]
ser = lambda i: ' '.join(str(int(round(v * 99))) for v in Fq[i, FEATS])
shots = np.concatenate([rng.choice(tr[y[tr] == c], 2, replace=False) for c in range(10)])
rng.shuffle(shots)
sys_msg = ('You are a network intrusion detection system. Each NetFlow record is given as 20 numeric '
           'features (percentile ranks 0-99). Classify the record into one of these classes: ' +
           '; '.join(f'{LET[c]} = {CLASSES[c]}' for c in range(10)) + '. Answer with the single letter only.')
ex = '\n'.join(f'Record: {ser(i)}\nClass: {LET[y[i]]}' for i in shots)
path = f'{OUT}/llm/qwen15'
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32); model.eval()
prefix = tok.apply_chat_template([{'role': 'system', 'content': sys_msg},
                                  {'role': 'user', 'content': 'Labelled examples:\n' + ex + '\n\nNow classify the following record.\nRecord: '}],
                                 tokenize=False, add_generation_prompt=False)
# strip the trailing end-of-turn so we can continue the user message
prefix = prefix.rsplit('<|im_end|>', 1)[0]
pid = tok(prefix, return_tensors='pt').input_ids
letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LET]
t0 = time.time()
with torch.no_grad():
    out = model(pid, use_cache=True); cache = out.past_key_values
t_prefix = time.time() - t0
print('prefix tokens', pid.shape[1], 'prefix s', t_prefix, flush=True)
preds, times = [], []
for i in qidx:
    suffix = ser(i) + '<|im_end|>\n<|im_start|>assistant\n'
    sid = tok(suffix, return_tensors='pt', add_special_tokens=False).input_ids
    ts = time.time()
    with torch.no_grad():
        c2 = copy.deepcopy(cache)
        o = model(sid, past_key_values=c2, use_cache=True)
    logits = o.logits[0, -1, letter_ids]
    preds.append(int(logits.argmax())); times.append(time.time() - ts)
    print(len(preds), LET[preds[-1]], LET[y[i]], round(times[-1], 2), flush=True)
preds = np.array(preds); yt = y[qidx]
from sklearn.metrics import f1_score
res = {'model': 'Qwen2.5-1.5B-Instruct (local CPU, fp32, 4 threads)', 'n_flows': len(qidx), 'n_shots': len(shots),
       'prefix_tokens': int(pid.shape[1]), 'prefix_s': t_prefix, 'per_flow_s_mean': float(np.mean(times)),
       'total_s': float(t_prefix + np.sum(times)),
       'macro_f1': float(f1_score(yt, preds, average='macro', labels=range(10), zero_division=0)),
       'bin_f1': float(f1_score(yt != BENIGN, preds != BENIGN)), 'acc': float((yt == preds).mean()),
       'pred_counts': np.bincount(preds, minlength=10).tolist()}
# majority-letter sanity: compare to models on the same flows
pos = {g: k for k, g in enumerate(emb['gid'])}
pm = emb['pred'][[pos[g] for g in qidx]]
res['dids_mfl_same_flows'] = {'macro_f1': float(f1_score(yt, pm, average='macro', labels=range(10), zero_division=0)),
                              'bin_f1': float(f1_score(yt != BENIGN, pm != BENIGN)), 'acc': float((yt == pm).mean())}
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(200, n_jobs=2, random_state=0, class_weight='balanced_subsample').fit(D['msg'][tr], y[tr])
ts = time.time(); pr = rf.predict(D['msg'][qidx]); t_rf = time.time() - ts
res['rf_same_flows'] = {'macro_f1': float(f1_score(yt, pr, average='macro', labels=range(10), zero_division=0)),
                        'bin_f1': float(f1_score(yt != BENIGN, pr != BENIGN)), 'acc': float((yt == pr).mean()), 'total_s': t_rf}
# a 30-shot RF on exactly the in-context examples (same information as the LLM)
rf30 = RandomForestClassifier(200, random_state=0).fit(Fq[shots], y[shots]); p30 = rf30.predict(Fq[qidx])
res['rf_on_prompt_shots'] = {'macro_f1': float(f1_score(yt, p30, average='macro', labels=range(10), zero_division=0)),
                             'bin_f1': float(f1_score(yt != BENIGN, p30 != BENIGN)), 'acc': float((yt == p30).mean())}
res['features_used'] = FEATS.tolist()
print(res)
save_json(res, f'{OUT}/llm/llm_results.json')
np.savez(f'{OUT}/llm/llm_preds.npz', qidx=qidx, pred=preds, y=yt, times=np.array(times))
