#!/usr/bin/env bash

python main.py \
  --mode \
    do_train \
  --data_path \
    SabaPivot/hate_speech \
  --model_name \
    monologg/kobert \
  --epochs \
    12 \
  --batch_size \
    32 \
  --lr \
    5e-5\
  --warmup_steps \
    20 \
  --save_limit 1 \
  --run_name \
    "your_run_name" \
  --data_type \
    "your_data_type" \
  --early_stopping_patience \
    5 \
