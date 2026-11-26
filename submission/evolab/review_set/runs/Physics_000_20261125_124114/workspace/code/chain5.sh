#!/bin/bash
cd /workspace/code
until [ "$(grep -c done ../outputs/growth_batch4.log 2>/dev/null)" -ge 3 ]; do sleep 30; done
python run_growth_exp.py "E2_d=0.0_T=0.22_tau=40_n=2" > ../outputs/growth_batch10.log 2>&1
