"""Extract experimental discharge data from NASA PCoE, CALCE CS2_36 and Oxford datasets.
Output: outputs/exp_data.pkl and outputs/data_summary.json"""
import os, glob, json, pickle
import numpy as np, pandas as pd, scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data'); O = os.path.join(ROOT, 'outputs')
out, summ = {}, {}

# ---------------- NASA ----------------
nasa_dir = os.path.join(D, 'NASA PCoE Dataset Repository', '1. BatteryAgingARC-FY08Q4')
out['NASA'] = {}
for cell in ['B0005', 'B0006', 'B0007', 'B0018']:
    m = sio.loadmat(os.path.join(nasa_dir, cell + '.mat'), squeeze_me=True, struct_as_record=False)
    cyc = [c for c in m[cell].cycle if c.type == 'discharge']
    lst = []
    for k, c in enumerate(cyc):
        d = c.data
        t = np.atleast_1d(d.Time).astype(float); I = -np.atleast_1d(d.Current_measured).astype(float)
        V = np.atleast_1d(d.Voltage_measured).astype(float); T = np.atleast_1d(d.Temperature_measured).astype(float)
        on = np.where(I > 1.0)[0]
        if len(on) < 10:
            continue
        i0, i1 = on[0], on[-1]
        lst.append(dict(cycle=k, t=t[i0:i1 + 1] - t[i0], V=V[i0:i1 + 1], T=T[i0:i1 + 1], I=I[i0:i1 + 1],
                        t_rest=t[i1 + 1:] - t[i1], V_rest=V[i1 + 1:], T_rest=T[i1 + 1:],
                        cap=float(np.atleast_1d(d.Capacity)[0]), T_amb=float(c.ambient_temperature),
                        Ion=float(np.median(I[i0:i1 + 1]))))
    out['NASA'][cell] = lst
    summ['NASA_' + cell] = dict(n_discharge=len(lst), cap_first=lst[0]['cap'], cap_last=lst[-1]['cap'],
                                I_mean=lst[0]['Ion'], dT_first=float(lst[0]['T'].max() - lst[0]['T'][0]),
                                t_first=float(lst[0]['t'][-1]))

# ---------------- CALCE CS2_36 ----------------
files = sorted(glob.glob(os.path.join(D, 'CS2_36', '*.xlsx')))
frames = []
for f in files:
    xl = pd.ExcelFile(f)
    sh = [s for s in xl.sheet_names if s.startswith('Channel')]
    df = pd.concat([xl.parse(s) for s in sh], ignore_index=True)
    df['__file'] = os.path.basename(f)
    frames.append((pd.to_datetime(df['Date_Time']).iloc[0], df))
frames.sort(key=lambda x: x[0])
cs2 = []
for _, df in frames:
    for cyc, g in df.groupby('Cycle_Index', sort=True):
        g = g[g['Step_Index'] == 7]
        if len(g) < 5:
            continue
        I = -g['Current(A)'].values
        if np.median(I) < 0.5:
            continue
        t = g['Step_Time(s)'].values.astype(float)
        cap = float(g['Discharge_Capacity(Ah)'].values[-1] - g['Discharge_Capacity(Ah)'].values[0])
        cs2.append(dict(file=g['__file'].iloc[0], cycle=int(cyc), t=t - t[0], V=g['Voltage(V)'].values.astype(float),
                        I=I, cap=cap))
out['CS2_36'] = cs2
summ['CS2_36'] = dict(n_discharge=len(cs2), cap_first=cs2[0]['cap'], cap_last=cs2[-1]['cap'],
                      I_mean=float(np.median(cs2[0]['I'])), t_first=float(cs2[0]['t'][-1]),
                      dt_sampling=float(np.median(np.diff(cs2[0]['t']))))

# ---------------- Oxford ----------------
m = sio.loadmat(os.path.join(D, 'Oxford Battery Degradation Dataset', 'ExampleDC_C1.mat'), squeeze_me=True,
                struct_as_record=False)
keys = [k for k in m if not k.startswith('__')]
ox = {}
for k in keys:
    s = m[k]
    for part in ['ch', 'dc']:
        p = getattr(s, part)
        ox[part] = dict(t=np.asarray(p.t, float), V=np.asarray(p.v, float), I=np.asarray(p.i, float) / 1000.0,
                        q=np.asarray(p.q, float) / 1000.0, T=np.asarray(p.T, float))
out['Oxford'] = ox
summ['Oxford'] = {part: dict(duration_s=float(ox[part]['t'][-1] - ox[part]['t'][0]), n=len(ox[part]['t']),
                             I_min=float(ox[part]['I'].min()), I_max=float(ox[part]['I'].max()),
                             q_range=float(np.ptp(ox[part]['q'])), T_mean=float(ox[part]['T'].mean()),
                             V_min=float(ox[part]['V'].min()), V_max=float(ox[part]['V'].max())) for part in ox}

pickle.dump(out, open(os.path.join(O, 'exp_data.pkl'), 'wb'))
json.dump(summ, open(os.path.join(O, 'data_summary.json'), 'w'), indent=2)
print(json.dumps(summ, indent=2))
