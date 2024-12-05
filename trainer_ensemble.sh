#!/usr/bin/env bash

python main.py \
  --mode \
    do_ensemble \
  --data_path \
    SabaPivot/hate_speech \
  --model_name \
    model1_name \
    model2_name \
  --model_dir \
    model1_path \
    model2_path \