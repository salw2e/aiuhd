#!/bin/sh
# sh bench/run.sh <data dir>
set -e
d=$1
for m in lanczos realesr-general-x4v3 realesrgan-x4plus bsrgan swinir-real; do
  [ -f $d/out/$m.json ] || python bench/run.py $d $d/out $m
done
echo done
