import pybamm, time, numpy as np, sys, warnings; warnings.filterwarnings('ignore')
def go(contact, extra, solver, teval):
    p=pybamm.ParameterValues("Ai2020"); j0=p["Negative electrode exchange-current density [A.m-2]"]
    upd={"Negative particle radius [m]":"[input]","Positive particle radius [m]":"[input]","Total heat transfer coefficient [W.m-2.K-1]":"[input]",
              "Negative electrode exchange-current density [A.m-2]":lambda ce,cs,cmax,T: pybamm.InputParameter("kn")*j0(ce,cs,cmax,T)}
    inp={"Negative particle radius [m]":5e-6,"Positive particle radius [m]":3e-6,"Total heat transfer coefficient [W.m-2.K-1]":20,"kn":1}
    if extra:
        upd['Initial concentration in negative electrode [mol.m-3]']=24108*pybamm.InputParameter('LLI'); inp['LLI']=1.0
        upd['Positive electrode active material volume fraction']=0.62*pybamm.InputParameter('LAM'); inp['LAM']=1.0
    if contact: upd['Contact resistance [Ohm]']=pybamm.InputParameter('Rf'); inp['Rf']=0.0
    p.update(upd,check_already_exists=False)
    m=pybamm.lithium_ion.SPMe(options={"thermal":"lumped","contact resistance":"true" if contact else "false"})
    sm=dict(m.default_submesh_types); sm["negative particle"]=pybamm.SymbolicUniform1DSubMesh; sm["positive particle"]=pybamm.SymbolicUniform1DSubMesh
    kw={} if solver is None else dict(solver=pybamm.IDAKLUSolver(rtol=1e-6,atol=1e-8))
    sim=pybamm.Simulation(m,parameter_values=p,submesh_types=sm,**kw)
    ts=[]
    for i in range(3):
        t0=time.time(); sol=sim.solve([0,5000] if teval is None else np.linspace(0,5000,501),inputs=inp); ts.append(round(time.time()-t0,3))
    print(contact,extra,solver,teval,ts, sim.solver.name if hasattr(sim.solver,'name') else sim.solver, flush=True)
a=[int(s) for s in sys.argv[1:]]
go(bool(a[0]),bool(a[1]),None if a[2]==0 else 1,None if a[3]==0 else 1)
