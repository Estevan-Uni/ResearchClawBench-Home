"""Re-implementation of the SimBA 'user-defined 16 body-part, 2 animal' feature family from pose only.
Column names follow the SimBA reference table so that features can be compared 1:1."""
import numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from common import BODYPARTS, FPS

PX_PER_MM = 4.2425  # estimated from reference table (distance features / px distance); assumed shared by both videos
WINDOWS = [2, 5, 6, 7.5, 15]  # SimBA: window = fps / w frames

def _xy(P, bp, a):
    return P[f'{bp}_{a}_x'].values.astype(float), P[f'{bp}_{a}_y'].values.astype(float)

def _d(P, bp1, a1, bp2, a2):
    x1, y1 = _xy(P, bp1, a1); x2, y2 = _xy(P, bp2, a2)
    return np.hypot(x1 - x2, y1 - y2) / PX_PER_MM

def _hull_perim(pts):
    try: return ConvexHull(pts).area
    except Exception: return 0.0

def simba_features(P, fps=FPS):
    F = {}
    arr = {a: np.stack([np.stack(_xy(P, bp, a), 1) for bp in BODYPARTS], 1) for a in (1, 2)}  # (n, 8, 2)
    for a in (1, 2):
        # SimBA uses scipy ConvexHull(...).area, which for 2-D points is the hull PERIMETER (kept for fidelity)
        F[f'Mouse_{a}_poly_area'] = np.array([_hull_perim(p) for p in arr[a]]) / PX_PER_MM
    for a in (1, 2):
        F[f'Mouse_{a}_nose_to_tail'] = _d(P, 'Nose', a, 'Tail_base', a)
    for a in (1, 2): F[f'Mouse_{a}_width'] = _d(P, 'Lat_left', a, 'Lat_right', a)
    for a in (1, 2): F[f'Mouse_{a}_Ear_distance'] = _d(P, 'Ear_left', a, 'Ear_right', a)
    for a in (1, 2): F[f'Mouse_{a}_Nose_to_centroid'] = _d(P, 'Nose', a, 'Center', a)
    for a in (1, 2): F[f'Mouse_{a}_Nose_to_lateral_left'] = _d(P, 'Nose', a, 'Lat_left', a)
    for a in (1, 2): F[f'Mouse_{a}_Nose_to_lateral_right'] = _d(P, 'Nose', a, 'Lat_right', a)
    for a in (1, 2): F[f'Mouse_{a}_Centroid_to_lateral_left'] = _d(P, 'Center', a, 'Lat_left', a)
    for a in (1, 2): F[f'Mouse_{a}_Centroid_to_lateral_right'] = _d(P, 'Center', a, 'Lat_right', a)
    F['Centroid_distance'] = _d(P, 'Center', 1, 'Center', 2)
    F['Nose_to_nose_distance'] = _d(P, 'Nose', 1, 'Nose', 2)
    F['M1_Nose_to_M2_lat_left'] = _d(P, 'Nose', 1, 'Lat_left', 2)
    F['M1_Nose_to_M2_lat_right'] = _d(P, 'Nose', 1, 'Lat_right', 2)
    F['M2_Nose_to_M1_lat_left'] = _d(P, 'Nose', 2, 'Lat_left', 1)
    F['M2_Nose_to_M1_lat_right'] = _d(P, 'Nose', 2, 'Lat_right', 1)
    F['M1_Nose_to_M2_tail_base'] = _d(P, 'Nose', 1, 'Tail_base', 2)
    F['M2_Nose_to_M1_tail_base'] = _d(P, 'Nose', 2, 'Tail_base', 1)
    S = P.shift(1).fillna(0)  # SimBA fills the shifted first frame with 0 (frame-0 movement artefact, kept for fidelity)
    mv_names = {'Center': 'centroid', 'Nose': 'nose', 'Tail_base': 'tail_base', 'Tail_end': 'tail_end',
                'Ear_left': 'left_ear', 'Ear_right': 'right_ear', 'Lat_left': 'lateral_left', 'Lat_right': 'lateral_right'}
    for bp, nm in mv_names.items():
        for a in (1, 2):
            x1, y1 = _xy(P, bp, a); x0, y0 = _xy(S, bp, a)
            F[f'Movement_mouse_{a}_{nm}'] = np.hypot(x1 - x0, y1 - y0) / PX_PER_MM
    for a in (1, 2):
        pa = pd.Series(F[f'Mouse_{a}_poly_area'])
        F[f'Mouse_{a}_polygon_size_change'] = (pa.shift(1).fillna(0) - pa).values
    for a in (1, 2):
        D = np.linalg.norm(arr[a][:, :, None, :] - arr[a][:, None, :, :], axis=-1) / PX_PER_MM
        iu = np.triu_indices(8, 1); v = D[:, iu[0], iu[1]]
        F[f'M{a}_largest_euclidean_distance_hull'] = v.max(1)
        F[f'M{a}_smallest_euclidean_distance_hull'] = v.min(1)
        F[f'M{a}_mean_euclidean_distance_hull'] = v.mean(1)
        F[f'M{a}_sum_euclidean_distance_hull'] = v.sum(1)
    F['Sum_euclidean_distance_hull_M1_M2'] = F['M1_sum_euclidean_distance_hull'] + F['M2_sum_euclidean_distance_hull']
    F['Total_movement_centroids'] = F['Movement_mouse_1_centroid'] + F['Movement_mouse_2_centroid']
    F['Total_movement_tail_ends'] = F['Movement_mouse_1_tail_end'] + F['Movement_mouse_2_tail_end']
    for a in (1, 2):
        F[f'Total_movement_all_bodyparts_M{a}'] = sum(F[f'Movement_mouse_{a}_{nm}'] for nm in mv_names.values())
    F['Total_movement_all_bodyparts_both_mice'] = F['Total_movement_all_bodyparts_M1'] + F['Total_movement_all_bodyparts_M2']
    F = pd.DataFrame(F, index=P.index)
    roll_src = [('Sum_euclid_distances_hull', 'Sum_euclidean_distance_hull_M1_M2'), ('Movement', 'Total_movement_centroids'),
                ('Distance', 'Centroid_distance'), ('Mouse1_width', 'Mouse_1_width'), ('Mouse2_width', 'Mouse_2_width'),
                ('Mouse1_mean_euclid_distances', 'M1_mean_euclidean_distance_hull'), ('Mouse2_mean_euclid_distances', 'M2_mean_euclidean_distance_hull'),
                ('Mouse1_smallest_euclid_distances', 'M1_smallest_euclidean_distance_hull'), ('Mouse2_smallest_euclid_distances', 'M2_smallest_euclidean_distance_hull'),
                ('Mouse1_largest_euclid_distances', 'M1_largest_euclidean_distance_hull'), ('Mouse2_largest_euclid_distances', 'M2_largest_euclidean_distance_hull'),
                ('Total_movement_all_bodyparts_both_mice', 'Total_movement_all_bodyparts_both_mice'), ('Total_movement_centroids', 'Total_movement_centroids'),
                ('Tail_base_movement_M1', 'Movement_mouse_1_tail_base'), ('Tail_base_movement_M2', 'Movement_mouse_2_tail_base'),
                ('Centroid_movement_M1', 'Movement_mouse_1_centroid'), ('Centroid_movement_M2', 'Movement_mouse_2_centroid'),
                ('Tail_end_movement_M1', 'Movement_mouse_1_tail_end'), ('Tail_end_movement_M2', 'Movement_mouse_2_tail_end'),
                ('Nose_movement_M1', 'Movement_mouse_1_nose'), ('Nose_movement_M2', 'Movement_mouse_2_nose')]
    new = {}
    for pre, src in roll_src:
        for w in WINDOWS:
            n = max(int(fps / w), 1); r = F[src].rolling(n, min_periods=1)
            new[f'{pre}_median_{w}'] = r.median(); new[f'{pre}_mean_{w}'] = r.mean(); new[f'{pre}_sum_{w}'] = r.sum()
    te = {a: F[f'Movement_mouse_{a}_tail_end'] - (F[f'Movement_mouse_{a}_tail_base'] + F[f'Movement_mouse_{a}_centroid'] + F[f'Movement_mouse_{a}_nose']) for a in (1, 2)}
    new['Tail_end_relative_to_tail_base_centroid_nose'] = te[1]
    for w in WINDOWS:
        n = max(int(fps / w), 1)
        new[f'Tail_end_relative_to_tail_base_centroid_nose_M1_{w}'] = te[1].rolling(n, min_periods=1).mean()
        new[f'Tail_end_relative_to_tail_base_centroid_nose_M2_mean_{w}'] = te[2].rolling(n, min_periods=1).mean()
    for a in (1, 2):
        ax, ay = _xy(P, 'Nose', a); bx, by = _xy(P, 'Center', a); cx, cy = _xy(P, 'Tail_base', a)
        ang = np.degrees(np.arctan2(cy - by, cx - bx) - np.arctan2(ay - by, ax - bx)); ang = np.where(ang < 0, ang + 360, ang)
        new[f'Mouse_{a}_angle'] = pd.Series(ang, index=P.index)
    new['Total_angle_both_mice'] = new['Mouse_1_angle'] + new['Mouse_2_angle']
    for w in WINDOWS:
        new[f'Total_angle_both_mice_{w}'] = new['Total_angle_both_mice'].rolling(max(int(fps / w), 1), min_periods=1).sum()
    F = pd.concat([F, pd.DataFrame(new, index=P.index)], axis=1)
    dev = {}
    dev_src = [('Total_movement_all_bodyparts_both_mice_deviation', 'Total_movement_all_bodyparts_both_mice'),
               ('Sum_euclid_distances_hull_deviation', 'Sum_euclidean_distance_hull_M1_M2'),
               ('M1_smallest_euclid_distances_hull_deviation', 'M1_smallest_euclidean_distance_hull'),
               ('M1_largest_euclid_distances_hull_deviation', 'M1_largest_euclidean_distance_hull'),
               ('M1_mean_euclid_distances_hull_deviation', 'M1_mean_euclidean_distance_hull'),
               ('Centroid_distance_deviation', 'Centroid_distance'), ('Total_angle_both_mice_deviation', 'Total_angle_both_mice'),
               ('Movement_mouse_1_deviation_centroid', 'Movement_mouse_1_centroid'), ('Movement_mouse_2_deviation_centroid', 'Movement_mouse_2_centroid'),
               ('Mouse_1_polygon_deviation', 'Mouse_1_poly_area'), ('Mouse_2_polygon_deviation', 'Mouse_2_poly_area')]
    for pre in ['Total_movement_all_bodyparts_both_mice_mean', 'Sum_euclid_distances_hull_mean', 'Mouse1_smallest_euclid_distances_mean',
                'Mouse1_largest_euclid_distances_mean', 'Mouse1_mean_euclid_distances_mean', 'Movement_mean', 'Distance_mean', 'Total_angle_both_mice']:
        for w in WINDOWS: dev_src.append((f'{pre}_{w}_deviation', f'{pre}_{w}'))
    for n, s in dev_src: dev[n] = F[s].mean() - F[s]  # SimBA convention: mean - value
    dv = pd.DataFrame(dev, index=P.index); F = pd.concat([F, dv], axis=1)
    pr = {}
    pr_src = [('Movement_percentile_rank', 'Total_movement_centroids'), ('Distance_percentile_rank', 'Centroid_distance'),
              ('Movement_mouse_1_percentile_rank', 'Movement_mouse_1_centroid'), ('Movement_mouse_2_percentile_rank', 'Movement_mouse_2_centroid'),
              ('Movement_mouse_1_deviation_percentile_rank', 'Movement_mouse_1_deviation_centroid'),
              ('Movement_mouse_2_deviation_percentile_rank', 'Movement_mouse_2_deviation_centroid'),
              ('Centroid_distance_percentile_rank', 'Centroid_distance'), ('Centroid_distance_deviation_percentile_rank', 'Centroid_distance_deviation')]
    for n, s in pr_src: pr[n] = F[s].rank(pct=True)
    # In the reference SimBA output the rolling-window '*_percentile_rank' columns equal the '*_deviation' columns (SimBA quirk, reproduced)
    for pre in ['Total_movement_all_bodyparts_both_mice_mean', 'Sum_euclid_distances_hull_mean', 'Mouse1_mean_euclid_distances_mean',
                'Mouse1_smallest_euclid_distances_mean', 'Mouse1_largest_euclid_distances_mean', 'Movement_mean', 'Distance_mean']:
        for w in WINDOWS: pr[f'{pre}_{w}_percentile_rank'] = F[f'{pre}_{w}_deviation']
    F = pd.concat([F, pd.DataFrame(pr, index=P.index)], axis=1)
    ex = {}
    a1 = pd.Series(F['Mouse_1_angle'])
    for w in WINDOWS:  # tortuosity: rolling sum of absolute heading-angle change of M1 (approximation)
        ex[f'Tortuosity_Mouse1_{w}'] = a1.diff().abs().fillna(0).rolling(max(int(fps / w), 1), min_periods=1).sum()
    pc = [f'{bp}_{a}_p' for a in (1, 2) for bp in BODYPARTS]
    sp = P[pc].sum(1); ex['Sum_probabilities'] = sp; ex['Sum_probabilities_deviation'] = sp.mean() - sp
    ex['Sum_probabilities_deviation_percentile_rank'] = ex['Sum_probabilities_deviation'].rank(pct=True)
    ex['Sum_probabilities_percentile_rank'] = ex['Sum_probabilities_deviation'].rank(pct=True)  # SimBA ranks the deviation
    for t in (0.1, 0.5, 0.75): ex[f'Low_prob_detections_{t}'] = (P[pc] < t).sum(1)
    F = pd.concat([F, pd.DataFrame(ex, index=P.index)], axis=1)
    return F.replace([np.inf, -np.inf], np.nan).fillna(0)

def tag_feature(name):
    """Return (category, role) for a feature name. Role: M1 (assumed resident), M2 (assumed intruder), inter (both / relational)."""
    n = name
    if n in ('Feature_1', 'Feature_2'): return 'index_control', 'none'
    if n.endswith(('_x', '_y')) and any(n.startswith(bp) for bp in BODYPARTS): cat = 'position'
    elif n.endswith('_p') and any(n.startswith(bp) for bp in BODYPARTS): cat = 'tracking'
    elif 'prob' in n.lower(): cat = 'tracking'
    elif any(k in n for k in ('Simon', 'zone')): cat = 'roi'
    elif 'angle' in n or 'Tortuosity' in n: cat = 'angle'
    elif any(k in n for k in ('Movement', 'movement', 'size_change', 'Tail_end_relative')): cat = 'movement'
    elif n.startswith('Distance') or any(k in n for k in ('Centroid_distance', 'Nose_to_nose', 'M1_Nose_to_M2', 'M2_Nose_to_M1')): cat = 'distance'
    else: cat = 'shape'
    # role
    if cat == 'position' or cat == 'tracking' and n.endswith('_p'):
        role = 'M1' if '_1_' in n else 'M2'
    elif cat == 'distance': role = 'inter'
    elif any(k in n for k in ('Mouse_1', 'Mouse1', 'M1_', '_M1', 'mouse_1')) and not any(k in n for k in ('Mouse_2', 'Mouse2', 'M2_', '_M2', 'mouse_2', 'both', 'M1_M2')):
        role = 'M1'
    elif any(k in n for k in ('Mouse_2', 'Mouse2', 'M2_', '_M2', 'mouse_2')) and not any(k in n for k in ('both', 'M1_M2')):
        role = 'M2'
    else: role = 'inter'  # combined (both animals) or relational
    if n.startswith('Tail_end_relative_to_tail_base_centroid_nose'): role = 'M2' if 'M2' in n else 'M1'
    return cat, role
