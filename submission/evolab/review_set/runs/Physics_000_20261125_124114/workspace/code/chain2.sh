#!/bin/bash
cd /workspace/code
alive() { for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null); case "$c" in "python run_growth_exp.py E1_d=0_T=0.22_tau=60_n=10"*) return 0;; esac; done; return 1; }
while alive; do sleep 30; done
python run_growth_exp.py "E3c_T=0.22_tau=60_n=3" > ../outputs/growth_batch4.log 2>&1
