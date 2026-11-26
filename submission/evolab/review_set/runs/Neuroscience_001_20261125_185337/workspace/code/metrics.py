"""Summary metrics comparing model predictions to experimentally known properties
(flyvis.utils.groundtruth_utils)."""
import numpy as np
from flyvis.utils import groundtruth_utils as gt
import fastsim as fs

TYPES = fs.central_type_order()
KNOWN_POL = {k: v for k, v in gt.polarity.items() if v != 0}
KNOWN_POL_NOPR = {k: v for k, v in KNOWN_POL.items() if not k.startswith("R")}
T4 = ["T4a", "T4b", "T4c", "T4d"]; T5 = ["T5a", "T5b", "T5c", "T5d"]
PD_TRUE = {k: np.deg2rad(v) for k, v in gt.preferred_directions.items()}


def ang_dist(a, b):
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return d


def summarise(fri, dsi, pd_, tuning_corr, intensity=(0, 1)):
    """fri (n_types,), dsi/pd (2, n_types) ordered by intensity (0=OFF,1=ON)."""
    intensity = list(intensity)
    on, off = intensity.index(1), intensity.index(0)
    ti = {t: i for i, t in enumerate(TYPES)}
    s = {}
    corr_all = [np.sign(fri[ti[t]]) == p for t, p in KNOWN_POL.items()]
    corr_np = [np.sign(fri[ti[t]]) == p for t, p in KNOWN_POL_NOPR.items()]
    s["fri_n_correct_all"] = int(np.sum(corr_all)); s["fri_n_known_all"] = len(corr_all)
    s["fri_acc_all"] = float(np.mean(corr_all)); s["fri_acc_noPR"] = float(np.mean(corr_np))
    fk = np.array([fri[ti[t]] for t in KNOWN_POL]); pk = np.array(list(KNOWN_POL.values()))
    s["fri_corr_known"] = float(np.corrcoef(fk, pk)[0, 1]) if np.std(fk) > 0 else np.nan
    # DSI correlation to known (max over intensities), flyvis definition
    kt = gt.known_dsi_types
    d = np.array([np.nanmax(dsi[:, ti[t]]) for t in kt]); g = np.array([1.0 if t in gt.motion_tuning else 0.0 for t in kt])
    s["dsi_corr_known"] = float(np.corrcoef(d, g)[0, 1]) if np.std(d) > 0 else np.nan
    s["dsi_T4_on"] = float(np.mean([dsi[on, ti[t]] for t in T4]))
    s["dsi_T5_off"] = float(np.mean([dsi[off, ti[t]] for t in T5]))
    s["dsi_T4_off"] = float(np.mean([dsi[off, ti[t]] for t in T4]))
    s["dsi_T5_on"] = float(np.mean([dsi[on, ti[t]] for t in T5]))
    s["dsi_nonDS_max"] = float(np.mean([np.nanmax(dsi[:, ti[t]]) for t in gt.no_motion_tuning]))
    s["pd_err_T4_deg"] = float(np.rad2deg(np.mean([ang_dist(pd_[on, ti[t]], PD_TRUE[t]) for t in T4])))
    s["pd_err_T5_deg"] = float(np.rad2deg(np.mean([ang_dist(pd_[off, ti[t]], PD_TRUE[t]) for t in T5])))
    tc = np.asarray(tuning_corr)
    if tc.shape == (2, 8):
        s["tuning_corr_T4"] = float(np.nanmean(tc[on, :4])); s["tuning_corr_T5"] = float(np.nanmean(tc[off, 4:]))
    return s
