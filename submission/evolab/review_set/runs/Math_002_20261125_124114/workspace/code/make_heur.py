"""Hand-designed (non-learned) action-scoring policy in the same network format: score ~ linear in features."""
import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from marl import Policy, NF
coef = np.zeros(NF)
coef[20] = -3.0   # excess collision-to-go of the action (background DP)
coef[21] = -1.0   # excess time-to-go
coef[22] = 1.0    # action is the single-agent optimum
coef[4] = -3.0    # team-mate already claimed target
coef[5] = -3.0    # swap with team-mate
coef[6] = -0.5    # undecided team-mate currently there
coef[24] = -1.5   # undecided team-mate intends to move to v
coef[25] = -1.0   # team-mates intend to pass v soon
coef *= 2.0
p = Policy(H=32); s = 0.05
p.W1[:] = 0; p.b1[:] = 0; p.w2[:] = 0
for q in range(NF):
    p.W1[q, q] = s; p.w2[q] = coef[q] / s
p.b2 = 0.0
p.save(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'policy_heur.npz'))
print('saved')
