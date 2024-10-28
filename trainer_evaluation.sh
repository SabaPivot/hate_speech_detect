#!/usr/bin/env bash

python main.py \
  --mode do_eval \
  --data_path SabaPivot/hate_speech \
  --model_name \
  monologg/koelectra-base-v3-discriminator \
  klue/bert-base \
  --model_dir \
  /home/careforme.dropout/huggingface_trainer/Augmented_5e-05_32_50_12_classifier_dropout=0.1_fp16_koelectra/checkpoint-295 \
  /home/careforme.dropout/huggingface_trainer/Augmented_0.0001_32_50_12_classifier_dropout=0.2_fp16_klue/bert/checkpoint-295 \
