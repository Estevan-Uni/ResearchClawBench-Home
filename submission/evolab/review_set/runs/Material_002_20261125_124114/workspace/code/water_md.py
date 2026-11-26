"""Exp. 1: 32 H2O, 12 A cubic box, Langevin NVT 330 K, dt 0.5 fs, friction 0.01/fs, 2000 steps, float32."""
import sys, time, json, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import get_calc, OUT
from ase import Atoms, units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from scipy.spatial.transform import Rotation as R
L = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
tag = sys.argv[3] if len(sys.argv) > 3 else "L12"
rng = np.random.default_rng(42)
mol = np.array([[0, 0, 0.119262], [0, 0.763239, -0.477047], [0, -0.763239, -0.477047]])
pos, sym = [], []
n = 4; a = L / n
for i in range(n):
    for j in range(n):
        for k in range(n):
            if (i + j + k) % 2: continue  # 32 sites of a 4x4x4 grid (fcc-like sublattice)
            c = (np.array([i, j, k]) + 0.5) * a
            rot = R.random(random_state=int(rng.integers(1e9))).as_matrix()
            pos += list(c + mol @ rot.T); sym += ["O", "H", "H"]
atoms = Atoms(sym, positions=pos, cell=[L] * 3, pbc=True)
assert len(atoms) == 96
atoms.calc = get_calc("float32")
MaxwellBoltzmannDistribution(atoms, temperature_K=330, rng=np.random.default_rng(1)); Stationary(atoms)
dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=330, friction=0.01 / units.fs, rng=np.random.default_rng(2))
traj = Trajectory(f"{OUT}/water_{tag}.traj", "w", atoms)
log = open(f"{OUT}/water_{tag}_log.csv", "w"); log.write("step,time_fs,T_K,Epot_eV,Ekin_eV,wall_s\n")
t0 = time.time()
def wr():
    s = dyn.nsteps
    log.write(f"{s},{s*0.5:.1f},{atoms.get_temperature():.2f},{atoms.get_potential_energy():.5f},{atoms.get_kinetic_energy():.5f},{time.time()-t0:.1f}\n"); log.flush()
    if s % 5 == 0: traj.write(atoms)
dyn.attach(wr, interval=1)
wr_density = 32 * 18.015 / 6.022e23 / (L * 1e-8) ** 3
json.dump({"L": L, "density_g_cm3": wr_density, "nsteps": NSTEP}, open(f"{OUT}/water_{tag}_meta.json", "w"))
dyn.run(NSTEP)
print("FINISHED", time.time() - t0)
