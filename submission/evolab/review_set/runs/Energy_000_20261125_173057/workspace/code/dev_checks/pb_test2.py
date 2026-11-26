import pybamm, time, numpy as np
print(list(pybamm.parameter_sets.keys()))
for ps in ["Ramadass2004","Ai2020","Marquis2019"]:
    try:
        p=pybamm.ParameterValues(ps)
        print(ps, p["Nominal cell capacity [A.h]"], p.get("Cell volume [m3]"), p.get("Cell cooling surface area [m2]"), p.get("Total heat transfer coefficient [W.m-2.K-1]"), p["Lower voltage cut-off [V]"],p["Upper voltage cut-off [V]"])
    except Exception as e: print(ps,"ERR",e)
# input test radius
p=pybamm.ParameterValues("Chen2020")
p.update({"Negative particle radius [m]":"[input]","Positive particle radius [m]":"[input]","Total heat transfer coefficient [W.m-2.K-1]":"[input]"})
m=pybamm.lithium_ion.SPMe(options={"thermal":"lumped"})
sim=pybamm.Simulation(m,parameter_values=p)
try:
    t0=time.time(); sol=sim.solve([0,4000],inputs={"Negative particle radius [m]":5e-6,"Positive particle radius [m]":5e-6,"Total heat transfer coefficient [W.m-2.K-1]":10}); print("ok",time.time()-t0, sol["Voltage [V]"].entries[-1])
    t0=time.time(); sol=sim.solve([0,4000],inputs={"Negative particle radius [m]":9e-6,"Positive particle radius [m]":3e-6,"Total heat transfer coefficient [W.m-2.K-1]":20}); print("ok2",time.time()-t0, sol["Time [s]"].entries[-1])
except Exception as e: print("radius input ERR",repr(e)[:500])
