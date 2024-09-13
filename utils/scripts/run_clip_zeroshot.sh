#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi



torchrun --master_port 1239 --nproc_per_node=8 \
    clip_zeroshot.py --config ${config}  ${@:3}