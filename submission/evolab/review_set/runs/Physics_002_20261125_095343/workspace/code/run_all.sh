#!/bin/bash
# Full reproduction (approx. 25 min on 2 cores). Run from the workspace root.
set -e
python code/load_data.py            # parse, dedup, audit -> outputs/audit_table.csv, data_cache.pkl
python code/inspect_mb.py           # MB / d24 construction checks
python code/estimators.py           # per-instance linXEB/logXEB/PT-MLE/MB/TRB + circuit bootstrap
python code/synthetic_validation.py # estimator bias / coverage on synthetic data
python code/model_analysis.py       # pooled PT-MLE, TRB fits, gate-count model + MC band, regressions, mirror inference
python code/validate_mps.py         # MPS discarded-weight fidelity vs exact overlaps (N<=16)
python code/classical_mps.py depth & python code/classical_mps.py nscan; wait
python code/classical_scrambling_tw.py
python code/classical_gap_summary.py
python code/figures.py
