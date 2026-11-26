#!/bin/bash
# usage: run_synthA.sh start end chunk
cd /workspace/code
for ((s=$1; s<$2; s+=$3)); do
  e=$((s+$3))
  if [ ! -f ../outputs/synth_A/A_${s}_${e}.done ]; then
    PYTHONHASHSEED=0 python synth.py A $s $e ../outputs/synth_A && touch ../outputs/synth_A/A_${s}_${e}.done
  fi
done
