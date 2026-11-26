#!/bin/bash
# kill processes whose cmdline contains $1 (python processes only), excluding this shell
pat="$1"
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null)
  case "$c" in python*"$pat"*) echo "kill $pid ${c:0:60}"; kill $pid;; esac
done
