import pybamm, time, numpy as np, warnings; warnings.filterwarnings('ignore')
p=pybamm.ParameterValues("Ai2020")
p.update({"Negative particle radius [m]":"[input]","Total heat transfer coefficient [W.m-2.K-1]":"[input]"})
inp={"Negative particle radius [m]":5e-6,"Total heat transfer coefficient [W.m-2.K-1]":20}
m=pybamm.lithium_ion.SPMe(options={"thermal":"lumped"})
sm=dict(m.default_submesh_types); sm["negative particle"]=pybamm.SymbolicUniform1DSubMesh
for mode in ["adaptive","interp","outvars"]:
    kw={}
    if mode=="outvars": kw=dict(solver=pybamm.IDAKLUSolver(output_variables=["Voltage [V]","Volume-averaged cell temperature [K]","Time [s]"]))
    sim=pybamm.Simulation(m,parameter_values=p,submesh_types=sm,**kw)
    ts=[]
    for i in range(3):
        t0=time.time()
        sol=sim.solve([0,5000],inputs=inp) if mode=="adaptive" else sim.solve([0,5000],inputs=inp,t_interp=np.arange(0,5000,2.0))
        V=sol["Voltage [V]"].entries; T=sol["Volume-averaged cell temperature [K]"].entries
        ts.append(round(time.time()-t0,3))
    print(mode,ts,len(V),sol.t[-1],flush=True)
