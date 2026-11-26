"""Prepare unbound docking partners from the Docking Benchmark entries for 1BRS:
barnase 1A2P (chain A) and barstar 1A19 (chain A, C82A). Barstar C40 is truncated to ALA so that the
unbound barstar matches the C40A/C82A pseudo-wild-type present in the 1BRS crystal (and SKEMPI).
altloc A kept, waters/HETATM/H removed, barstar renamed chain D (as in the reference)."""
import sys
raw, out = sys.argv[1], sys.argv[2]
def prep(fn, chain, newchain, outfn, ala=()):
    keep = []
    for l in open(fn):
        if not l.startswith("ATOM") or l[21] != chain or l[16] not in " A" or l[76:78].strip() == "H":
            continue
        resid = int(l[22:26]); name = l[12:16].strip()
        if resid in ala:
            if name not in ("N", "CA", "C", "O", "CB"):
                continue
            l = l[:17] + "ALA" + l[20:]
        l = l[:16] + " " + l[17:21] + newchain + l[22:]
        keep.append(l)
    open(outfn, "w").write("".join(keep) + "END\n")
    print(outfn, len(keep), "atoms")
prep(f"{raw}/1A2P.pdb", "A", "A", f"{out}/barnase_unbound.pdb")
prep(f"{raw}/1A19.pdb", "A", "D", f"{out}/barstar_unbound.pdb", ala=(40,))
