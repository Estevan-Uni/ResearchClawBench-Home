#!/bin/bash
# Matched-budget ablations of the sequence-frame model (same LR schedule as the 800-step main run, stopped at 200).
cd /workspace/code
for ab in no_trunk no_ref no_bonds; do
  python train.py --name seq_$ab --canon --canon_mode seq --$ab --steps 800 --stop_at 200 --diff_batch 6 --threads 1 --save_at 200 > /workspace/outputs/train_seq_$ab.log 2>&1
done
echo ABLATIONS_DONE > /workspace/outputs/ablations_done.flag
