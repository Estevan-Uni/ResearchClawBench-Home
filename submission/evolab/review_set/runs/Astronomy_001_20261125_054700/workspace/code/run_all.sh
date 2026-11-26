#!/bin/bash
# Full reproducible pipeline (≈3 min). Requires: numpy scipy matplotlib pandas camb>=1.5 getdist
set -e
cd "$(dirname "$0")"
python common.py              # parse & print inputs
python make_chains.py         # Gaussian mock chains + validation  -> outputs/chains, outputs/chain_validation.csv
python plot_triangle.py       # figure3_triangle*.png, figure3b_ede_params.png, figure_param_shifts.png
python propagate.py           # CAMB on chain sub-samples, sigma8 check -> outputs/propagation_summary.json
python distances.py           # fiducial tests + figure6_distance.png  (needs curves from propagate.py)
python acoustic.py            # figure_acoustic_plane.png
python overview_and_summary.py  # figure1_data_overview.png, outputs/chi2_summary.csv
python convention_test.py     # sign-convention sensitivity
