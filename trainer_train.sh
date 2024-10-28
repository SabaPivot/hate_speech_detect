#!/usr/bin/env bash

python main.py \
  --mode do_train \
  --data_path SabaPivot/hate_speech \
  --model_name monologg/kobert \
  --epochs 12 \
  --batch_size 32 \
  --lr 5e-5\
  --warmup_steps 20 \
  --save_limit 1 \
  --run_name "Augmented_1e-4_32_20_12_classifier_dropout=0.1_fp16_monologg/kobert" \
  --data_type "Augmented" \
  --early_stopping_patience 5 \
