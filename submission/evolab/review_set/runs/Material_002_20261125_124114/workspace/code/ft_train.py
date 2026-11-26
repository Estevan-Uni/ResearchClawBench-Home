"""Few-shot fine-tuning of MACE-MP-0b3-medium on PBE/6-31G* data for Rxn 1 (custom PyTorch loop).
usage: ft_train.py N mode(ft|scratch) steps lr tag"""
import sys, os, json, copy, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import MODEL, OUT
import torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float32)
from mace import data as mdata
from mace.tools import utils, torch_geometric
from ase.io import read
N, mode, steps, lr, tag = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
torch.manual_seed(0); rng = np.random.default_rng(0)
model = torch.load(MODEL, weights_only=False, map_location="cpu").float()
zt = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers]); rc = float(model.r_max)
data = read(f"{OUT}/ft_data_rxn1.extxyz", ":")
train_pool = [a for a in data if a.info["set"] == "train"]; test = [a for a in data if a.info["set"] == "test"]
idx = rng.permutation(len(train_pool))[:N]; train = [train_pool[i] for i in idx]
def to_batch(a):
    ad = mdata.AtomicData.from_config(mdata.config_from_atoms(a), z_table=zt, cutoff=rc, heads=["default"])
    return next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1, shuffle=False))).to_dict()
def predict(m, atoms_list):
    E, F = [], []
    for a in atoms_list:
        out = m(to_batch(a), training=False, compute_force=True)
        E.append(float(out["energy"].detach()[0])); F.append(out["forces"].detach().numpy())
    return np.array(E), F
def metrics(m, atoms_list):
    E, F = predict(m, atoms_list)
    Er = np.array([a.get_potential_energy() for a in atoms_list]); Fr = [a.get_forces() for a in atoms_list]
    dE = E - Er; dEc = dE - dE.mean()
    frmse = float(np.sqrt(np.mean(np.concatenate([(f - g).ravel() ** 2 for f, g in zip(F, Fr)]))))
    neb = [i for i, a in enumerate(atoms_list) if a.info["kind"] == "neb"]
    prof = (E[neb] - E[neb][0]).tolist() if neb else None
    return {"F_rmse_meV_A": 1000 * frmse, "E_rel_rmse_meV": 1000 * float(np.sqrt(np.mean(dEc ** 2))), "E_offset_eV": float(dE.mean()),
            "E_rel_rmse_meV_atom": 1000 * float(np.sqrt(np.mean(dEc ** 2))) / len(atoms_list[0]), "neb_profile": prof}
if mode == "scratch":
    with torch.no_grad():
        for p in model.parameters():
            s = float(p.std()) if p.numel() > 1 else 1.0
            p.copy_(torch.randn_like(p) * (s if s > 0 else 1.0))
# E0 alignment: single per-atom constant shift for H and C (VASP/PAW vs all-electron Gaussian referencing)
E0, _ = predict(model, train)
delta = float(np.mean([(a.get_potential_energy() - e) / len(a) for a, e in zip(train, E0)]))
with torch.no_grad():
    ae = model.atomic_energies_fn.atomic_energies
    for z in (1, 6): ae[..., zt.z_to_index(z)] += delta
res = {"N": N, "mode": mode, "steps": steps, "lr": lr, "E0_shift_eV_per_atom": delta, "test_zero_shot_aligned": metrics(model, test)}
for p in model.parameters(): p.requires_grad_(True)
opt = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
batches = [(to_batch(a), torch.tensor(a.get_potential_energy(), dtype=torch.float32), torch.tensor(a.get_forces(), dtype=torch.float32), len(a)) for a in train]
B = min(2, N); hist = []; t0 = time.time()
model.train()
for st in range(steps):
    opt.zero_grad(); L = 0.0
    for j in rng.choice(len(batches), B, replace=False):
        b, er, fr, n = batches[j]
        out = model(b, training=True, compute_force=True)
        loss = 10 * ((out["energy"][0] - er) / n) ** 2 + torch.mean((out["forces"] - fr) ** 2)
        (loss / B).backward(); L += float(loss) / B
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step(); sched.step(); hist.append(L)
    if st % 25 == 0 or st == steps - 1: print(st, round(L, 5), round(time.time() - t0, 1), flush=True)
model.eval()
res["train_loss"] = hist; res["test"] = metrics(model, test); res["train"] = metrics(model, train); res["wall_s"] = time.time() - t0
print(json.dumps({k: v for k, v in res.items() if k not in ("train_loss",)}, default=str)[:1500])
json.dump(res, open(f"{OUT}/ft_{tag}.json", "w"), indent=1)
if len(sys.argv) > 6 and sys.argv[6] == "save":
    for p in model.parameters(): p.requires_grad_(False)
    torch.save(model, f"{OUT}/model/ft_{tag}.model")
