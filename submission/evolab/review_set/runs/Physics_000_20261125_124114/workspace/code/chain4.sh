#!/bin/bash
cd /workspace/code
alive() { for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null); case "$c" in "python run_growth_exp.py E3b_d=0.09"*) return 0;; esac; done; return 1; }
while alive; do sleep 30; done
python run_one.py "E2b_d=0.0_T=0.22_tau=30_n=1_N=212" 0 > ../outputs/growth_batch9.log 2>&1
