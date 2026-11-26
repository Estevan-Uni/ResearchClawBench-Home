import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pyscf import gto, dft, lib
lib.num_threads(1)
from ase.calculators.calculator import Calculator, all_changes
HA, BOHR = 27.211386245988, 0.529177210903
def pbe(atoms, basis="6-31g*", grad=False, spin=None):
    nel = int(sum(atoms.numbers)); spin = (nel % 2) if spin is None else spin
    mol = gto.M(atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)], basis=basis, spin=spin, unit="Angstrom", verbose=0)
    mf = (dft.UKS(mol) if spin else dft.RKS(mol)).density_fit(); mf.xc = "pbe"; mf.grids.level = 2; mf.conv_tol = 1e-9; mf.max_cycle = 200
    e = mf.kernel()
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel()
    if grad:
        g = mf.nuc_grad_method().kernel(); return e * HA, -g * HA / BOHR
    return e * HA
class PySCFPBE(Calculator):
    implemented_properties = ["energy", "forces"]
    def __init__(self, basis="6-31g*", **kw):
        super().__init__(**kw); self.basis = basis
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f = pbe(self.atoms, self.basis, grad=True)
        self.results = {"energy": e, "forces": f}
