#!/bin/bash
# Second half of the pipeline (re-run after fixing the gen-file key bug); waits for Short+ to finish.
set -x
cd /workspace/code
L=/workspace/outputs/logs; CK=/workspace/outputs/ckpt
until [ -f $L/ALL_TRAIN_DONE ]; do sleep 20; done
python train.py --stage medium --init $CK/short.pt --out $CK/medium.pt --curriculum 2:250,4:150 --lr 1e-4 --bs 8 --warmup 50 --offset 20 --gen /workspace/outputs/gen/short_day5.npz --minutes 40 > $L/train_medium.log 2>&1
python gen.py medium > $L/gen_medium.log 2>&1
python train.py --stage long   --init $CK/medium.pt --out $CK/long.pt  --curriculum 2:250,4:150 --lr 1e-4 --bs 8 --warmup 50 --offset 40 --gen /workspace/outputs/gen/medium_day10.npz --minutes 40 > $L/train_long.log 2>&1
python forecast.py cascade short_only > $L/forecast2.log 2>&1
touch $L/CASCADE_DONE2
