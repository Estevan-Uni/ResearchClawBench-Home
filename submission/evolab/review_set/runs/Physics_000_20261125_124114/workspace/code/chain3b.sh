#!/bin/bash
cd /workspace/code
until [ -f "../outputs/growth/E2b_d=0.08_T=0.22_tau=30_n=2_N=192_s0.json" ]; do sleep 20; done
for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null); case "$c" in "python run_growth_exp.py E2b_d=0.08"*) kill ${p#/proc/};; esac; done
python run_one.py "E2b_d=0.06_T=0.22_tau=30_n=2_N=192" 1 > ../outputs/growth_batch7.log 2>&1
