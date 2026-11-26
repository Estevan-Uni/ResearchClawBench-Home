import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, IMG
from ase.io import read
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# NOTE: tags *_N54 used the full training pool (48 configs; N is capped at the pool size)
runs = {t: json.load(open(f"{OUT}/ft_{t}.json")) for t in ["ft_N5", "ft_N20", "ft_N54", "scratch_N54"]}
data = read(f"{OUT}/ft_data_rxn1.extxyz", ":"); neb = [a for a in data if a.info["kind"] == "neb"]
Epbe = np.array([a.get_potential_energy() for a in neb]); Epbe -= Epbe[0]
zs = runs["ft_N5"]["test_zero_shot_aligned"]
summ = {"n_train_pool": sum(a.info["set"] == "train" for a in data), "n_test": sum(a.info["set"] == "test" for a in data),
        "zero_shot": {k: zs[k] for k in ["F_rmse_meV_A", "E_rel_rmse_meV", "E_rel_rmse_meV_atom"]}, "pbe_profile_on_zs_path": Epbe.tolist()}
for t, r in runs.items():
    summ[t] = {k: r["test"][k] for k in ["F_rmse_meV_A", "E_rel_rmse_meV", "E_rel_rmse_meV_atom"]}; summ[t]["train_F_rmse"] = r["train"]["F_rmse_meV_A"]
    summ[t]["neb_profile"] = r["test"]["neb_profile"]; summ[t]["max_on_zs_path"] = max(r["test"]["neb_profile"]); summ[t]["wall_s"] = r["wall_s"]; summ[t]["E0_shift"] = r["E0_shift_eV_per_atom"]
summ["pbe_max_on_zs_path"] = float(Epbe.max()); summ["zs_max_on_zs_path"] = max(zs["neb_profile"])
if os.path.exists(f"{OUT}/neb_rxn1_ft.json"): summ["neb_ft"] = json.load(open(f"{OUT}/neb_rxn1_ft.json"))
if os.path.exists(f"{OUT}/barriers_pbe_ft.json"): summ["pbe_on_ft_path"] = json.load(open(f"{OUT}/barriers_pbe_ft.json"))["rxn1"]
json.dump(summ, open(f"{OUT}/ft_summary.json", "w"), indent=1); print(json.dumps({k: v for k, v in summ.items() if k not in ("neb_ft",)}, indent=0)[:3000])
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
Ns = [5, 20, 54]; Neff = {5: 5, 20: 20, 54: summ["n_train_pool"]}
ax[0].plot([Neff[n] for n in Ns], [summ[f"ft_N{n}"]["F_rmse_meV_A"] for n in Ns], "o-", color="#1f5f8b", label="fine-tuned MACE-MP-0b3")
ax[0].axhline(zs["F_rmse_meV_A"], color="#c0392b", ls="--", label="zero-shot (E0-aligned)")
ax[0].scatter([Neff[54]], [summ["scratch_N54"]["F_rmse_meV_A"]], color="#e67e22", marker="s", s=50, zorder=3, label="from scratch (same arch., same steps)")
ax[0].set_xlabel("N training configurations"); ax[0].set_ylabel("test force RMSE (meV/Å)"); ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[1].plot([Neff[n] for n in Ns], [summ[f"ft_N{n}"]["E_rel_rmse_meV"] for n in Ns], "o-", color="#1f5f8b"); ax[1].axhline(zs["E_rel_rmse_meV"], color="#c0392b", ls="--")
ax[1].scatter([Neff[54]], [summ["scratch_N54"]["E_rel_rmse_meV"]], color="#e67e22", marker="s", s=50, zorder=3)
ax[1].set_xlabel("N training configurations"); ax[1].set_ylabel("test relative-energy RMSE (meV / 10-atom config)"); ax[1].set_yscale("log")
s = np.linspace(0, 1, len(Epbe))
ax[2].plot(s, Epbe, "k-o", ms=4, label="PBE/6-31G* (reference)")
ax[2].plot(s, zs["neb_profile"], "--", color="#c0392b", label="zero-shot MACE-MP-0b3")
for n, c in zip(Ns, ["#9ecae1", "#4292c6", "#08306b"]): ax[2].plot(s, summ[f"ft_N{n}"]["neb_profile"], "-", color=c, label=f"fine-tuned N={Neff[n]}")
ax[2].plot(s, summ["scratch_N54"]["neb_profile"], ":", color="#e67e22", label=f"scratch N={Neff[54]}")
ax[2].set_xlabel("image along zero-shot CI-NEB path"); ax[2].set_ylabel("E − E(img 0) (eV)"); ax[2].legend(fontsize=7); ax[2].set_title("Held-out energy profile (Rxn 1)", fontsize=10)
plt.tight_layout(); plt.savefig(f"{IMG}/finetune_learning.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for t, c in zip(runs, ["#9ecae1", "#4292c6", "#08306b", "#e67e22"]):
    L = np.array(runs[t]["train_loss"]); k = 10; ax.plot(np.convolve(L, np.ones(k) / k, "valid"), color=c, label=t)
ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss (running mean)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{IMG}/finetune_loss.png", dpi=150); plt.close()
