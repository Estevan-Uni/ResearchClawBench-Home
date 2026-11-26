#!/bin/bash
# Reproduce all outputs in order (seeds fixed in common.py: SEED=20241125; mod_sim SEED+1; train_sim SEED+2)
set -e
cd "$(dirname "$0")"
python audit.py > ../outputs/audit.log
python pore_models.py > ../outputs/pore_models.log
python performance.py > ../outputs/performance.log
python m6a_eval.py > ../outputs/m6a_eval.log
python signal_sim.py > ../outputs/signal_sim.log
python mod_sim.py > ../outputs/mod_sim.log
python train_sim.py > ../outputs/train_sim.log
echo DONE > ../outputs/run_all.done
