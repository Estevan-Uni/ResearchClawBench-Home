import pybamm, time, numpy as np
for ps in ["Marquis2019","Chen2020"]:
    p = pybamm.ParameterValues(ps)
    keys=[k for k in p.keys() if any(s in k for s in ["particle radius","exchange-current","heat transfer","Nominal cell capacity","cut-off","Ambient","Initial temperature","Cell volume","specific heat"])]
    print(ps, {k:(p[k] if not callable(p[k]) else 'fn') for k in keys})
    model = pybamm.lithium_ion.SPMe(options={"thermal":"lumped"})
    t0=time.time()
    sim=pybamm.Simulation(model,parameter_values=p,experiment=pybamm.Experiment(["Discharge at 1C until 3.0 V","Rest for 30 minutes"]) if False else None)
    cap=p["Nominal cell capacity [A.h]"]
    sim=pybamm.Simulation(model,parameter_values=p)
    sol=sim.solve([0,4000])
    t=sol["Time [s]"].entries; V=sol["Voltage [V]"].entries; T=sol["X-averaged cell temperature [K]"].entries
    print(ps,"time",time.time()-t0,"tend",t[-1],"V",V[0],V[-1],"dT",T.max()-T[0])
    t0=time.time(); sol=sim.solve([0,4000]); print("2nd solve",time.time()-t0)
