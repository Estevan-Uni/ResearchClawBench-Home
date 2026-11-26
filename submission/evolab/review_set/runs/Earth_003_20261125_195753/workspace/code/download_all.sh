#!/bin/bash
# Sequential download queue (network ~0.6-3 MB/s, resets parallel connections).
cd /workspace/code
L=/workspace/outputs/logs
python download_clim.py static >> $L/dl_clim.log 2>&1
python download_era5.py 2017 >> $L/dl_era5.log 2>&1
python download_era5.py 2020 >> $L/dl_era5.log 2>&1
NT=12 python download_benchmarks.py ens >> $L/dl_bench.log 2>&1
python download_era5.py 2016 >> $L/dl_era5.log 2>&1
python download_clim.py >> $L/dl_clim.log 2>&1
python download_era5.py 2015 >> $L/dl_era5.log 2>&1
NT=12 python download_benchmarks.py hres >> $L/dl_bench.log 2>&1
python download_era5.py 2014 >> $L/dl_era5.log 2>&1
NT=12 python download_benchmarks.py fuxi >> $L/dl_bench.log 2>&1
python download_era5.py 2013 2012 2011 2010 >> $L/dl_era5.log 2>&1
touch /workspace/outputs/logs/DOWNLOAD_DONE
