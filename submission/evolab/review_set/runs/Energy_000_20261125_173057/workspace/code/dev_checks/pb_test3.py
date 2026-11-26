import pybamm, time, numpy as np
for ps in ["Ai2020","Chen2020"]:
    p=pybamm.ParameterValues(ps)
    j0=p["Negative electrode exchange-current density [A.m-2]"]
    p.update({"Negative particle radius [m]":"[input]","Positive particle radius [m]":"[input]","Total heat transfer coefficient [W.m-2.K-1]":"[input]",
              "Negative electrode exchange-current density [A.m-2]":lambda ce,cs,cmax,T: pybamm.InputParameter("kn")*j0(ce,cs,cmax,T)}, check_already_exists=False)
    m=pybamm.lithium_ion.SPMe(options={"thermal":"lumped"})
    sm=dict(m.default_submesh_types); sm["negative particle"]=pybamm.SymbolicUniform1DSubMesh; sm["positive particle"]=pybamm.SymbolicUniform1DSubMesh
    I=p["Nominal cell capacity [A.h]"]
    sim=pybamm.Simulation(m,parameter_values=p,submesh_types=sm)
    for inp in [(p_:=None)]: pass
    for Rn,Rp,k,h in [(5.86e-6,5.22e-6,1,10),(1e-5,3e-6,0.3,20),(3e-6,1e-5,3,5)]:
        try:
            t0=time.time(); sol=sim.solve([0,5000],inputs={"Negative particle radius [m]":Rn,"Positive particle radius [m]":Rp,"Total heat transfer coefficient [W.m-2.K-1]":h,"kn":k})
            T=sol["X-averaged cell temperature [K]"].entries
            print(ps,"ok %.3f s tend %.0f V0 %.3f Vend %.3f dT %.2f"%(time.time()-t0, sol["Time [s]"].entries[-1], sol["Voltage [V]"].entries[0],sol["Voltage [V]"].entries[-1],T.max()-T[0]))
        except Exception as e: print(ps,"ERR",repr(e)[:400])
    print(p["Initial concentration in negative electrode [mol.m-3]"], p["Maximum concentration in negative electrode [mol.m-3]"], p["Current function [A]"])
