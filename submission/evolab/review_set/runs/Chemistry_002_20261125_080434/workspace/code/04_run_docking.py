"""Step 4: generate and run HADDOCK3 docking workflows for each restraint scenario.

Workflow (per scenario):
  topoaa -> rigidbody (it0) -> caprieval -> seletop -> flexref (it1, semi-flexible SA)
         -> emref (water/EM refinement analogue) -> caprieval
Clustering (FCC) is done afterwards in Python (code/05_analyse_docking.py) because the compiled
contact_fcc helper could not be built in this environment (no C compiler).

usage: python code/04_run_docking.py [scenario ...]
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/outputs/docking"
RES = f"{ROOT}/outputs/restraints"

# name: (input set, restraint scenario, rigidbody sampling, n refined)
RUNS = {
    "unbound_hotspot": ("unbound", "hotspot", 200, 20),
    "unbound_ideal": ("unbound", "ideal", 200, 20),
    "unbound_noisy": ("unbound", "noisy", 200, 20),
    "unbound_onesided": ("unbound", "onesided", 200, 20),
    "unbound_abinitio": ("unbound", "abinitio", 400, 20),
    "bound_hotspot": ("bound", "hotspot", 100, 10),
    "bound_abinitio": ("bound", "abinitio", 400, 20),
}


def cfg(name, inp, scen, nsamp, nref):
    mols = (["inputs/barnase_unbound.pdb", "inputs/barstar_unbound.pdb"] if inp == "unbound"
            else ["inputs/barnase.pdb", "inputs/barstar.pdb"])
    if scen == "abinitio":
        rb = "cmrest = true\nsurfrest = true\n"
        ref = ""
    else:
        rb = f'ambig_fname = "{RES}/{scen}.tbl"\n'
        ref = rb
    return f'''run_dir = "run_{name}"
mode = "local"
ncores = 2
molecules = {mols!r}

[topoaa]

[rigidbody]
{rb}sampling = {nsamp}

[caprieval]
reference_fname = "inputs/ref_1brs.pdb"

[seletop]
select = {nref}

[flexref]
{ref}
[emref]
{ref}
[caprieval]
reference_fname = "inputs/ref_1brs.pdb"
'''.replace("'", '"')


if __name__ == "__main__":
    names = sys.argv[1:] or list(RUNS)
    os.chdir(D)
    for n in names:
        if os.path.exists(f"run_{n}/6_caprieval/capri_ss.tsv"):
            print("skip", n); continue
        open(f"{n}.cfg", "w").write(cfg(n, *RUNS[n]))
        subprocess.run(["rm", "-rf", f"run_{n}"])
        t = time.time()
        with open(f"{n}.log", "w") as fh:
            subprocess.run(["haddock3", f"{n}.cfg"], stdout=fh, stderr=subprocess.STDOUT)
        print(n, "done in %.1f min" % ((time.time() - t) / 60), flush=True)
