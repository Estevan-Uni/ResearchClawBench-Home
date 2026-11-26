"""Stimulus generation (flyvis-native datasets) and response -> metric pipeline that
re-uses the *native* flyvis metric functions (flash_response_index,
direction_selectivity_index, preferred_direction, correlation_to_known_tuning_curves)
on xarray datasets formatted exactly as flyvis.analysis.stimulus_responses does."""
import sys
sys.path.insert(0, "/workspace/code")
import numpy as np
import xarray as xr
import torch
import fastsim as fs
from flyvis.datasets.flashes import Flashes
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis import flash_responses as FR
from flyvis.analysis import moving_bar_responses as MB

SPEEDS_ALL = (2.4, 4.8, 9.7, 13, 19, 25)
FLASH_CFG = {'dynamic_range': [0, 1], 't_stim': 1, 't_pre': 1.0, 'dt': 1 / 200, 'radius': (-1, 6),
             'alternations': (0, 1, 0)}


def edge_cfg(speeds=SPEEDS_ALL, dt=1 / 200):
    return {'offsets': (-10, 11), 'intensities': [0, 1], 'speeds': tuple(speeds), 'height': 80,
            'post_pad_mode': "continue", 'dt': dt, 'device': 'cpu', 't_pre': 1.0, 't_post': 1.0}


_cache = {}


def flash_stimuli(dt=1 / 200):
    key = ("flash", dt)
    if key not in _cache:
        cfg = dict(FLASH_CFG, dt=dt)
        ds = Flashes(**cfg)
        stim = np.stack([np.asarray(ds[i]).reshape(-1, 721) for i in range(len(ds))])
        _cache[key] = (stim.astype(np.float32), ds.arg_df.copy(), cfg)
    return _cache[key]


def edge_stimuli(speeds=SPEEDS_ALL, dt=1 / 200):
    key = ("edge", tuple(speeds), dt)
    if key not in _cache:
        cfg = edge_cfg(speeds, dt)
        ds = MovingEdge(**{k: v for k, v in cfg.items()})
        stim = np.stack([np.asarray(ds[i]).reshape(-1, 721) for i in range(len(ds))])
        _cache[key] = (stim.astype(np.float32), ds.arg_df.copy(), cfg)
    return _cache[key]


def _time(n_frames, dt, t_pre):
    return np.arange(n_frames).astype(float) * dt - t_pre


def to_dataset(responses, arg_df, cfg, types=None):
    """responses: (S, T, n_types) central responses (one model) -> xr.Dataset in flyvis format
    with network_id dimension of size 1."""
    types = types or fs.central_type_order()
    S, nT, _ = responses.shape
    ds = xr.Dataset({'responses': (['network_id', 'sample', 'frame', 'neuron'], responses[None])},
                    coords={'sample': np.arange(S),
                            **{col: ('sample', arg_df[col].values) for col in arg_df.columns}})
    ds.coords.update({'frame': np.arange(nT), 'neuron': np.arange(len(types)),
                      'time': ('frame', _time(nT, cfg['dt'], cfg.get('t_pre', 0.0))),
                      'cell_type': ('neuron', np.array(types)), 'network_id': ('network_id', [0])})
    ds.attrs['config'] = cfg
    return ds


def simulate_flashes(sim, dt=1 / 200, V_grey=None):
    stim, arg_df, cfg = flash_stimuli(dt)
    r = sim.simulate(stim, dt=dt, V0=V_grey)
    return r, arg_df, cfg


def simulate_edges(sim, speeds=SPEEDS_ALL, dt=1 / 200, V_grey=None, record="central"):
    """Trimmed but numerically equivalent simulation of flyvis moving-edge responses.
    For each speed: the leading grey frames are integrated once with batch size 1, then
    the batch is integrated only until the end of the flyvis peak-response time mask
    (+5 frames); remaining frames are filled with NaN (never used by native metrics)."""
    stim, arg_df, cfg = edge_stimuli(speeds, dt)
    S, nT, _ = stim.shape
    time = _time(nT, dt, cfg['t_pre'])
    n_out = len(fs.topo().central_cells_index) if (isinstance(record, str) and record == "central") else (
        fs.topo().n_nodes if isinstance(record, str) else len(record))
    out = np.full((S, nT, n_out), np.nan, dtype=np.float32)
    if V_grey is None:
        V_grey = sim.steady_state(dt, t_pre=1.0)
    start, end = cfg['offsets']
    for speed in np.unique(arg_df.speed.values):
        idx = np.where(arg_df.speed.values == speed)[0]
        st = stim[idx]
        # number of leading grey frames identical across samples
        grey = np.all(np.abs(st - 0.5) < 1e-7, axis=(0, 2))
        n_grey = int(np.argmin(grey)) if not grey.all() else nT
        t0, t1 = MB.time_window(speed, from_column=start * 2.25 / 5.8, to_column=(end - 1) * 2.25 / 5.8,
                                start=start, end=end)
        last = min(nT, int(np.searchsorted(time, t1, side="right")) + 5)
        # grey prefix (batch 1)
        if n_grey > 0:
            rpre, V = sim.simulate(st[:1, :n_grey], dt=dt, V0=V_grey, record=record, return_last=True)
            out[idx, :n_grey] = rpre[0][None]
        else:
            V = V_grey
        r = sim.simulate(st[:, n_grey:last], dt=dt, V0=V, record=record)
        out[idx, n_grey:last] = r
    return out, arg_df, cfg


# ---------------------------------------------------------------- metrics ----
def fri(flash_resp, arg_df, cfg, radius=6):
    ds = to_dataset(flash_resp, arg_df, cfg)
    f = FR.flash_response_index(ds, radius=radius)
    return np.asarray(f.values).squeeze()  # (n_types,)


def edge_metrics(edge_resp, arg_df, cfg):
    ds = to_dataset(edge_resp, arg_df, cfg)
    dsi = MB.direction_selectivity_index(ds)  # dims (intensity, neuron) after squeeze
    pd_ = MB.preferred_direction(ds)
    peak = MB.peak_responses(ds)  # (network, sample, neuron)
    try:
        tc = MB.correlation_to_known_tuning_curves(ds)
        tc = tc.squeeze()
        tc = np.asarray(tc.transpose('intensity', 'neuron').values) if 'intensity' in tc.dims else np.asarray(tc.values)
    except Exception as e:  # pragma: no cover
        tc = np.full(8, np.nan)
    dsi_v = np.asarray(dsi.transpose('intensity', 'neuron').values)
    pd_v = np.asarray(pd_.squeeze().transpose('intensity', 'neuron').values)
    return dict(dsi=dsi_v, pd=pd_v, peak=np.asarray(peak.values)[0], tuning_corr=tc,
                intensity=np.asarray(dsi.intensity.values))
