#!/bin/bash
# launch sharded resumable runs: launch.sh METHOD NSHARDS
cd /workspace/code
for k in $(seq 0 $(($2-1))); do
  nohup python -u run_exp.py $1 all $k/$2 > ../outputs/logs/${1}_$k.txt 2>&1 &
done
