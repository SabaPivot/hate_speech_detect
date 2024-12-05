#!/usr/bin/env bash

python main.py \
  --mode do_eval \
  --data_path "SabaPivot/hate_speech" \
  --model_name team-lucid/deberta-v3-xlarge-korean team-lucid/deberta-v3-xlarge-korean team-lucid/deberta-v3-xlarge-korean \
  --model_dir \
    1e-05_team-lucid/deberta-v3-xlarge-korean/checkpoint-1896 \
    3e-05_team-lucid/deberta-v3-xlarge-korean/checkpoint-4424 \
    7e-06_team-lucid/deberta-v3-xlarge-korean/checkpoint-1896