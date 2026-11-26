"""Dual-encoder graph VAE for acid/epoxide vitrimer pairs (PyTorch).

- Two independent message-passing graph encoders (acid, epoxide) -> q(z_a|G_a), q(z_e|G_e)
- Two independent SELFIES GRU decoders p(S_a|z_a), p(S_e|z_e)
- A jointly trained MLP Tg head on z = [z_a, z_e]  (Gomez-Bombarelli-style property-shaped latent space)
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import selfies as sf
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

ELEM = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B']
BT = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
OXI = Chem.MolFromSmarts('[CH2]1[CH1,CH0]O1')
ANY_EPOX = Chem.MolFromSmarts('C1OC1')


def onehot(v, choices):
    o = [0] * (len(choices) + 1)
    o[choices.index(v) if v in choices else len(choices)] = 1
    return o


def atom_feat(a):
    return (onehot(a.GetSymbol(), ELEM) + onehot(a.GetDegree(), [0, 1, 2, 3, 4]) +
            onehot(a.GetTotalNumHs(), [0, 1, 2, 3]) + onehot(a.GetFormalCharge(), [-1, 0, 1]) +
            onehot(str(a.GetHybridization()), ['SP', 'SP2', 'SP3']) +
            [int(a.GetIsAromatic()), int(a.IsInRing()), int(a.IsInRingSize(3)), int(a.IsInRingSize(5)), int(a.IsInRingSize(6))])


NF = len(atom_feat(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))


def mol_graph(smi):
    m = Chem.MolFromSmiles(smi)
    x = np.array([atom_feat(a) for a in m.GetAtoms()], np.float32)
    src, dst, et = [], [], []
    for b in m.GetBonds():
        i, j, t = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), BT.get(b.GetBondType(), 0)
        src += [i, j]; dst += [j, i]; et += [t, t]
    return x, np.array(src, np.int64), np.array(dst, np.int64), np.array(et, np.int64)


def batch_graphs(graphs):
    xs, ss, ds, es, bs = [], [], [], [], []
    off = 0
    for g, (x, s, d, e) in enumerate(graphs):
        xs.append(x); ss.append(s + off); ds.append(d + off); es.append(e); bs.append(np.full(len(x), g))
        off += len(x)
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ss)),
            torch.from_numpy(np.concatenate(ds)), torch.from_numpy(np.concatenate(es)),
            torch.from_numpy(np.concatenate(bs)), len(graphs))


class Vocab:
    def __init__(self, token_lists):
        toks = sorted({t for tl in token_lists for t in tl})
        self.itos = ['<pad>', '<bos>', '<eos>'] + toks
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, tl, maxlen):
        ids = [1] + [self.stoi[t] for t in tl] + [2]
        return ids[:maxlen]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == 2: break
            if i > 2: out.append(self.itos[i])
        return ''.join(out)


def to_selfies_tokens(smi):
    try:
        s = sf.encoder(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
        return list(sf.split_selfies(s))
    except Exception:
        return None


def selfies_to_smiles(s):
    try:
        smi = sf.decoder(s)
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None and m.GetNumAtoms() > 0 else None
    except Exception:
        return None


class GraphEncoder(nn.Module):
    def __init__(self, nf=NF, h=128, L=4, zdim=32, nrel=4):
        super().__init__()
        self.inp = nn.Linear(nf, h)
        self.rel = nn.ModuleList([nn.Linear(h, h * nrel, bias=False) for _ in range(L)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h)) for _ in range(L)])
        self.norm = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.h, self.nrel = h, nrel
        self.readout = nn.Sequential(nn.Linear(2 * h, 2 * h), nn.SiLU())
        self.mu = nn.Linear(2 * h, zdim); self.lv = nn.Linear(2 * h, zdim)

    def forward(self, x, src, dst, et, batch, ng):
        h = Fn.silu(self.inp(x))
        for rel, upd, nrm in zip(self.rel, self.upd, self.norm):
            msg_all = rel(h).view(-1, self.nrel, self.h)       # all relation transforms in one matmul
            msg = msg_all[src, et]                             # pick relation per edge
            agg = torch.zeros_like(h).index_add_(0, dst, msg)
            h = nrm(h + upd(torch.cat([h, agg], -1)))
        s = torch.zeros(ng, self.h).index_add_(0, batch, h)
        cnt = torch.zeros(ng).index_add_(0, batch, torch.ones(len(batch))).clamp(min=1)[:, None]
        g = self.readout(torch.cat([s / 10.0, s / cnt], -1))
        return self.mu(g), self.lv(g).clamp(-8, 4)


class SelfiesDecoder(nn.Module):
    def __init__(self, V, zdim=32, emb=48, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.init = nn.Linear(zdim, h)
        self.gru = nn.GRU(emb + zdim, h, batch_first=True, num_layers=1)
        self.out = nn.Sequential(nn.Linear(h + zdim, h), nn.SiLU(), nn.Linear(h, V))
        self.V = V

    def forward(self, z, tgt):  # teacher forcing; tgt (B,T) incl. bos ... eos, pad=0
        inp = tgt[:, :-1]
        e = self.emb(inp)
        zz = z[:, None, :].expand(-1, inp.shape[1], -1)
        h0 = torch.tanh(self.init(z))[None]
        o, _ = self.gru(torch.cat([e, zz], -1), h0)
        logits = self.out(torch.cat([o, zz], -1))
        return Fn.cross_entropy(logits.reshape(-1, self.V), tgt[:, 1:].reshape(-1), ignore_index=0,
                                reduction='none').view(tgt.shape[0], -1).sum(1)

    @torch.no_grad()
    def generate(self, z, maxlen=120, temp=0.0):
        B = z.shape[0]
        h = torch.tanh(self.init(z))[None]
        tok = torch.ones(B, dtype=torch.long)
        out = []; done = torch.zeros(B, dtype=torch.bool)
        for t in range(maxlen):
            e = self.emb(tok)[:, None]
            o, h = self.gru(torch.cat([e, z[:, None]], -1), h)
            lg = self.out(torch.cat([o[:, 0], z], -1))
            lg[:, 0] = -1e9; lg[:, 1] = -1e9
            if temp <= 0:
                tok = lg.argmax(-1)
            else:
                tok = torch.multinomial(Fn.softmax(lg / temp, -1), 1)[:, 0]
            tok = torch.where(done, torch.zeros_like(tok), tok)
            out.append(tok); done = done | (tok == 2)
            if done.all(): break
        return torch.stack(out, 1).numpy()


class DualVAE(nn.Module):
    def __init__(self, Va, Ve, zdim=32):
        super().__init__()
        self.enc_a = GraphEncoder(zdim=zdim); self.enc_e = GraphEncoder(zdim=zdim)
        self.dec_a = SelfiesDecoder(Va, zdim); self.dec_e = SelfiesDecoder(Ve, zdim)
        self.head = nn.Sequential(nn.Linear(2 * zdim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.zdim = zdim

    def encode(self, ga, ge):
        return self.enc_a(*ga), self.enc_e(*ge)


def fg_check(smi, kind):
    """Functional-group correctness: acid = exactly 2 COOH & no epoxide; epoxide = exactly 2 terminal oxiranes & no COOH."""
    if smi is None: return False
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    na = len(m.GetSubstructMatches(ACID)); ne = len(m.GetSubstructMatches(OXI)); nany = len(m.GetSubstructMatches(ANY_EPOX))
    if kind == 'acid':
        return na == 2 and nany == 0
    return ne == 2 and nany == 2 and na == 0
