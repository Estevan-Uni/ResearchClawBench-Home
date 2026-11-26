"""Step 5: HADDOCK3 scoring of every SKEMPI 2.0 mutant of barnase-barstar (single + double, Ala and
non-Ala substitutions), mimicking the alascan protocol: side chain truncated to CB (CA for Gly),
residue renamed, missing atoms rebuilt by CNS topology generation, energy-minimised and scored with
`haddock3-score --full` (HADDOCK score = 1.0 Evdw + 0.2 Eelec + 1.0 Edesolv).

usage: python code/05_score_mutants.py TEMPLATE.pdb OUT.csv [nproc] [single]
(optional 4th argument "single" restricts the scan to single-point mutants)
"""
import os, sys, re, subprocess, tempfile, shutil
import pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AA3 = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU", G="GLY", H="HIS", I="ILE",
           L="LEU", K="LYS", M="MET", F="PHE", P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")


def mutate_lines(lines, muts):
    """muts: list like ['KA27A', 'DD39A'] (wt aa, chain, resid, mut aa)."""
    spec = {}
    for m in muts:
        wt, ch, resid, mt = m[0], m[1], int(m[2:-1]), m[-1]
        spec[(ch, resid)] = (AA3[wt], AA3[mt])
    out, seen = [], set()
    for l in lines:
        if not l.startswith("ATOM"):
            continue
        key = (l[21], int(l[22:26]))
        if key in spec:
            wt3, mt3 = spec[key]
            assert l[17:20] == wt3, (key, l[17:20], wt3)
            keep = ("N", "CA", "C", "O") if mt3 == "GLY" else ("N", "CA", "C", "O", "CB")
            if l[12:16].strip() not in keep:
                continue
            l = l[:17] + mt3 + l[20:]
            seen.add(key)
        out.append(l)
    assert seen == set(spec), (muts, seen)
    return out


def score(args):
    template, label, muts = args
    lines = [l for l in open(template) if l.startswith("ATOM")]
    tmp = tempfile.mkdtemp(prefix="mut_", dir=os.environ.get("MUT_TMP", "/tmp"))
    try:
        pdb = os.path.join(tmp, "model.pdb")
        open(pdb, "w").write("".join(mutate_lines(lines, muts) if muts else lines) + "END\n")
        r = subprocess.run(["haddock3-score", "model.pdb", "--full", "--run_dir", "run"],
                           cwd=tmp, capture_output=True, text=True)
        txt = r.stdout + r.stderr
        sc = float(re.search(r"HADDOCK-score \(emscoring\) = (\S+)", txt).group(1))
        comp = dict(re.findall(r"(vdw|elec|desolv|air|bsa)=([-\d.eE+]+)", txt))
        return dict(label=label, score=sc, **{k: float(v) for k, v in comp.items()})
    except Exception as e:
        return dict(label=label, error=str(e)[:200])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    template, outcsv = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    sk = pd.read_csv(f"{ROOT}/outputs/skempi_1brs.csv")
    if len(sys.argv) > 4 and sys.argv[4] == "single":
        sk = sk[sk.n_mut == 1]
    labels = ["WT"] + sorted(sk.mutations.unique())
    # template residue names may differ from crystal (e.g. docking models): keep all
    jobs = [(template, lab, [] if lab == "WT" else lab.split(",")) for lab in labels]
    with Pool(nproc) as p:
        res = p.map(score, jobs, chunksize=1)
    df = pd.DataFrame(res)
    wt = df[df.label == "WT"].iloc[0]
    for k in ["score", "vdw", "elec", "desolv", "bsa"]:
        df[f"d_{k}"] = df[k] - wt[k]  # mutant - WT (positive = destabilising for energies)
    df.to_csv(outcsv, index=False)
    print(df.head(), "\nerrors:", df.get("error", pd.Series(dtype=str)).notna().sum())
