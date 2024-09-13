#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

torchrun --master_port 1237 --nproc_per_node=4 \
    test.py --config ${config} --weights ${weight} ${@:3}