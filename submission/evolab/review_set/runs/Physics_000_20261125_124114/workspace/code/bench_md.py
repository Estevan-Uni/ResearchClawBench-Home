import numpy as np, time, icoshell as ic, md_growth as md
# identifier validation on ideal relaxed shells
seed=[(0,0),(0,1)]
idf=md.Identifier(seed,[1,1],1.0,[(0,2),(1,1)])
for c in [(0,2),(1,1)]:
    E,X1,D,I=md.relaxed_template(seed+[c],[1,1,1.0])
    rng=np.random.default_rng(1); from scipy.spatial.transform import Rotation as Rot
    Rr=Rot.random(random_state=3).as_matrix(); X2=X1@Rr.T
    print(c, idf.identify(X2,0,np.where(I==2)[0]))
seed=[(0,0),(0,1),(1,1)]
idf=md.Identifier(seed,[1,1.1,1.3],1.4,[(1,2),(2,1),(0,2)])
for c in [(1,2),(2,1)]:
    E,X1,D,I=md.relaxed_template(seed+[c],[1,1.1,1.3,1.4])
    print(c, idf.identify(X1@Rr.T,0,np.where(I==3)[0]))
# timing
E,X1,D,I=md.relaxed_template([(0,0),(0,1),(1,1),(1,2)],[1,1.1,1.3,1.42])
t=time.time()
r=md.run_growth(X1,D,1.46,5,tau_dep=5.0,anneal_tau=0,rng=0)
print('117 seed, 5 deposits x 1000 steps:',time.time()-t,'s ->',(time.time()-t)/5000*1e6,'us/step')
E,X1,D,I=md.relaxed_template([(0,0),(0,1)],[1,1])
t=time.time()
r=md.run_growth(X1,D,1.0,5,tau_dep=5.0,anneal_tau=0,rng=0)
print('13 seed:',(time.time()-t)/5000*1e6,'us/step')
