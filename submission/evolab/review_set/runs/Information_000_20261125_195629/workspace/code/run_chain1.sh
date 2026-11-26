#!/bin/bash
cd /workspace/code
python generate_t2i.py ../outputs/jobs_t2i_janus.json ../outputs/t2i > ../outputs/log_t2i_janus.txt 2>&1
python understand.py ../outputs/jobs_core.json ../outputs/understanding_core.jsonl > ../outputs/log_und_core.txt 2>&1
python probe_pathways.py > ../outputs/log_probe.txt 2>&1
python understand.py ../outputs/jobs_robust.json ../outputs/understanding_robust.jsonl > ../outputs/log_und_robust.txt 2>&1
echo CHAIN1_DONE >> ../outputs/log_und_robust.txt
