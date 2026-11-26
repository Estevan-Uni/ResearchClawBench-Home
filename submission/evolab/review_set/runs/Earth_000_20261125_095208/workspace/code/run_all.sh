#!/usr/bin/env bash
# Full reproducible pipeline (run from /workspace)
set -e
python code/inventory.py > outputs/inventory_log.txt
python code/calendar_weights.py
python code/run_consensus.py
python code/analysis.py > outputs/analysis_log.txt
python code/make_figures.py
python code/build_report.py
