#!/usr/bin/env bash

python main.py \
  --mode \
    do_eval \
  --data_path \
    SabaPivot/hate_speech \
  --model_name \
    monologg/koelectra-base-v3-discriminator \
    klue/bert-base \
  --model_dir \
    model1_path \
    model2_path \
