"""Produce markdown tables of the coupling bounds (GeV^-1) and lambda."""
import json, os, numpy as np
from data_overview import ROOT
d = json.load(open(os.path.join(ROOT, "outputs/coupling_results.json")))
def f(x): return "—" if (x is None or not np.isfinite(x)) else f"{x:.1e}"
L = ["| μ [eV] | α(M=15.7) | level-mixing c=1 (t_Sal) | mixing bracket c∈[1/25,25] | bosenova c0=5 (t_Sal) | bosenova bracket c0∈[0.5,50] | mixing c=1, t=3 Myr | mixing c=1, age-marg. | λ=μ²g² (mixing c=1) | λ (bosenova c0=5) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in d["table"]:
    mu = r["mu"]; al = 7.4828e9 * 15.67 * mu
    lam = lambda g: (mu * 1e-9 * g) ** 2 if np.isfinite(g) else np.nan
    L.append(f"| {mu:.2e} | {al:.3f} | {f(r['mixing_1_0'])} | {f(r['mixing_0.04_0'])} – {f(r['mixing_25_0'])} | {f(r['bosenova_5_0'])} | {f(r['bosenova_0.5_0'])} – {f(r['bosenova_50_0'])} | {f(r['mixing_1_1'])} | {f(r['mixing_1_marg'])} | {f(lam(r['mixing_1_0']))} | {f(lam(r['bosenova_5_0']))} |")
s = "\n".join(L); print(s)
open(os.path.join(ROOT, "outputs/coupling_table.md"), "w").write(s + "\n")
