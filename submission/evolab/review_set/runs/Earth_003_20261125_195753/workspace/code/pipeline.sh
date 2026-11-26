#!/bin/bash
# Full cascade training pipeline (run after pretraining). Stage order and time caps chosen for 2 CPU cores.
set -x
cd /workspace/code
L=/workspace/outputs/logs; CK=/workspace/outputs/ckpt
until [ -f $CK/pre_log.json ]; do sleep 20; done
python train.py --stage short  --init $CK/pre.pt   --out $CK/short.pt  --curriculum 2:350,4:200,6:100 --lr 1e-4 --bs 8 --warmup 50 --minutes 45 > $L/train_short.log 2>&1
python gen.py short 1000 > $L/gen_short.log 2>&1
python train.py --stage medium --init $CK/short.pt --out $CK/medium.pt --curriculum 2:250,4:150 --lr 1e-4 --bs 8 --warmup 50 --offset 20 --gen /workspace/outputs/gen/short_day5.npz --minutes 28 > $L/train_medium.log 2>&1
python gen.py medium > $L/gen_medium.log 2>&1
python train.py --stage long   --init $CK/medium.pt --out $CK/long.pt  --curriculum 2:250,4:150 --lr 1e-4 --bs 8 --warmup 50 --offset 40 --gen /workspace/outputs/gen/medium_day10.npz --minutes 28 > $L/train_long.log 2>&1
python forecast.py cascade short_only > $L/forecast.log 2>&1
touch /workspace/outputs/logs/CASCADE_DONE
python train.py --stage shortplus --init $CK/short.pt --out $CK/shortplus.pt --curriculum 2:500,4:300 --lr 1e-4 --bs 8 --warmup 50 --minutes 50 > $L/train_shortplus.log 2>&1
python forecast.py shortplus >> $L/forecast.log 2>&1
touch /workspace/outputs/logs/ALL_TRAIN_DONE
