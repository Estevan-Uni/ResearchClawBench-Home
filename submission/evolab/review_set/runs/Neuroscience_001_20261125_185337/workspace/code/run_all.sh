#!/bin/bash
# Sequential pipeline (2 CPU cores): wait for 01 validation, then run analyses.
export FLYVIS_ROOT_DIR=/workspace/outputs/flyvis_root
cd /workspace
while ! grep -q "edge_max_abs_err" outputs/01_validate.log; do sleep 10; done
sleep 20
python code/01b_validate_edges_metrics.py > outputs/01b.log 2>&1
python code/03_ensemble_responses.py > outputs/03.log 2>&1
python code/04_ablations.py > outputs/04.log 2>&1
python code/05_mechanism.py > outputs/05.log 2>&1
python code/07_flow.py > outputs/07.log 2>&1
echo PIPELINE_DONE >> outputs/pipeline.log
