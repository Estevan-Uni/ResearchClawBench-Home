#!/bin/bash
cd /workspace/code
until grep -q CHAIN1_DONE ../outputs/log_und_robust.txt 2>/dev/null; do sleep 20; done
python understand.py ../outputs/jobs_meme_extra.json ../outputs/understanding_core.jsonl > ../outputs/log_und_meme_extra.txt 2>&1
echo CHAIN3_DONE >> ../outputs/log_und_meme_extra.txt
