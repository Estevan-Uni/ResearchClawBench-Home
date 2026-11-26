#!/bin/bash
cd /workspace/code
until grep -q CHAIN1_DONE ../outputs/log_und_robust.txt 2>/dev/null; do sleep 20; done
python generate_t2i.py ../outputs/jobs_t2i_more_a.json ../outputs/t2i > ../outputs/log_t2i_more_a.txt 2>&1
python probe_encoders.py 12 > ../outputs/log_probe_enc.txt 2>&1
python generate_t2i.py ../outputs/jobs_t2i_more_b.json ../outputs/t2i > ../outputs/log_t2i_more_b.txt 2>&1
echo CHAIN2_DONE >> ../outputs/log_t2i_more_b.txt
