# Hate Speech Detection

A Korean hate speech detection model using transformer-based models with ensemble voting capabilities.

## Overview

This project implements a hate speech detection system that:
- Uses multiple transformer models (BERT, KoELECTRA, etc.)
- Supports model ensemble through soft voting
- Includes hate vocabulary-based rule adjustments
- Provides training, evaluation, and ensemble prediction capabilities

## Installation

1. Clone the repository:
```
git clone [repository-url]
cd hate_speech_detect
```

2. Install dependencies:
```
pip install torch transformers datasets wandb pytorch-lightning tqdm sklearn
```

## Usage

### Training

Run a single model training:
```
bash trainer_train.sh
```

Key training parameters:
- `--model_name`: Transformer model to use (e.g., monologg/kobert)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--data_type`: Type of training data
- `--early_stopping_patience`: Early stopping patience value

### Evaluation

Evaluate single or multiple models:
```
bash trainer_evaluation.sh
```

### Ensemble Prediction

Run ensemble prediction with multiple models:
```
bash trainer_ensemble.sh
```

## Project Structure
```
hate_speech_detect/
├── data.py                    # 데이터 준비 및 토큰화 구현
├── ensemble.py                # 앙상블 예측 구현
├── main.py                    # 메인 함수 및 인자 파싱 구현
├── model.py                   # 모델 훈련 및 추론 구현 
├── trainer_ensemble.sh        # 앙상블 예측 실행 스크립트
├── trainer_evaluation.sh      # 평가 실행 스크립트
├── trainer_train.sh           # 훈련 실행 스크립트
├── utils.py                   # compute metrics 구현
└── README.md
```
## Features

### Model Support
- 다양한 한국어 트랜스포머 모델 지원
- 설정 가능한 모델 매개변수
- FP16 학습 지원
- 조기 종료(Early stopping) 구현

### Ensemble Voting
- Soft voting mechanism
- Hate vocabulary rule-based adjustments
- Multiple model support

### Training Features
- Wandb를 통한 실험 추적
- 사용자 지정 학습률
- 그래디언트 누적
- 웜업이 포함된 코사인 학습률 스케줄러

## Data
본 프로젝트는 [국립 국어원의 인공지능(AI) 말평 - 혐오표현 탐지](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=56)가 제공한 데이터를 사용합니다. 

### Special Tokens
해당 데이터셋은 개인 정보 보호를 위해 다음과 같은 특수 토큰을 포함합니다. 아래의 특수 토큰을 모델의 tokenizer에 special_token으로 추가해야 원활한 학습이 가능합니다.
- &name&
- &affiliation&
- &social-security-num&
- &tel-num&
- &card-num&
- &bank-account&
- &num&
- &online-account&
