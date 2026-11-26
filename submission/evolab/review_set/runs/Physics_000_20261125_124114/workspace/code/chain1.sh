#!/bin/bash
cd /workspace/code
while grep -q "python scans.py LJ" /proc/[0-9]*/cmdline 2>/dev/null; do
  found=0
  for p in /proc/[0-9]*; do c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null); case "$c" in "python scans.py LJ"*) found=1;; esac; done
  [ $found = 0 ] && break
  sleep 20
done
python scans.py LJ c11x > ../outputs/scans_LJx.log 2>&1
python scans.py Morse c00 c01 c11 c02 c12 c03 c22 > ../outputs/scans_Morse.log 2>&1
python scans.py harm c00 c01 c11 c02 c12 c03 c22 > ../outputs/scans_harm.log 2>&1
python gupta.py > ../outputs/gupta.log 2>&1
