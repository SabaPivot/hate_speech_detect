# 혐오 발언 탐지
한국어 Encoder-only 모델을 사용한 혐오 발언 탐지 모델입니다. 국립 국어원의 [혐오 발언 탐지](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=56) 대회 참가를 위하여 작성한 코드입니다.

본 프로젝트 레포트는 [레포트 링크](https://drive.google.com/file/d/10y2J4i73C34Ook2Deq2C2hXyVflY_8fk/view)를 참고해주세요.

## 개요
본 repository는 다음과 같은 기능을 제공합니다:
- Encoder-only Transformer 모델을 위한 훈련 스크립트
- hard voting 지원 (model.py 참고)
- Soft voting 지원 (ensemble.py 참고)
- 훈련, 평가, 앙상블 예측 제공

## 설치

1. Clone repository:
```
git clone [repository-url]
cd hate_speech_detect
```

2. 필수 라이브러리 설치:
```
pip install -r requirements.txt
```

## 사용법  

### 훈련
```
bash trainer_train.sh
```

다음과 같은 parameter를 조정할 수 있습니다:
- `--model_name`: 사용할 모델의 huggingface repository 경로
- `--epochs`: 훈련 epoch 수
- `--batch_size`: 훈련 batch size
- `--lr`: 학습률 (다중 학습률 지원)
- `--data_type`: 훈련 데이터 타입
- `--early_stopping_patience`: Early stopping patience 값

### 평가

단일 모델 평가 실행:
```
bash trainer_evaluation.sh
```
최종 결과는 .jsonl 파일 형식으로 저장됩니다.

### 앙상블 기법 적용

다중 모델 앙상블 추론 실행:
```
bash trainer_ensemble.sh
```

다음과 같은 parameter를 조정할 수 있습니다:
- `model_dir`: 사용할 모델의 경로, 원하는 만큼 ensemble 적용할 모델을 추가할 수 있습니다.
```
  --model_dir \
    model1_path \
    model2_path \
    model3_path \
    model4_path \
    ...
```
최종 결과는 .jsonl 파일 형식으로 저장됩니다.

## 폴더 구조
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

## 데이터
본 프로젝트는 [국립 국어원의 인공지능(AI) 말평 - 혐오표현 탐지](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=56)가 제공한 데이터를 사용합니다. 

해당 사이트에서 데이터 사용 허가를 승인 받은 이후, huggingface library에 private dataset으로 추가한 이후 `data.py`의 아래 코드로 데이터를 불러오기를 권장합니다.
```
login(token="hf_token")
...
def prepare_datasets(args):
    datasets = load_dataset(args.data_path)
```

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

## 결과
| Model               | f1_score |
|---------------------|----------|
| xlm-roberta-base-1  | 0.900    |
| xlm-roberta-base-2  | 0.900    |
| ko-electra-base     | 0.862    |
| ko-bert-base        | 0.871    |
| **Hard Voting**     | **0.928**    |
| **Soft Voting**     | **0.921**    |


### rule-based 적용 결과
아래와 같이 특정 혐오 단어가 `hate_voca`에 포함되어 있지만, 모델이 혐오 표현으로 예측이 실패한 경우 예측 결과를 rule-based 방법으로 수정합니다.
```
    for i, data in enumerate(test_dataset):
        if any(word in data["input"] for word in hate_voca) and result_array[i] != 1:
            count += 1
            result_array[i] = 1
```

`Rule-based 적용 이후, 약 2,000 개의 전체 테스트 데이터 셋 중, 약 10개의 데이터를 더 정확하게 예측하였고, 이에 따라 f1 score가 약 0.3 ~ 0.6점 증가하였습니다.`

## 연락처
해당 프로젝트와 관련하여 질문이나 이슈가 있을 경우, github 이슈를 생성해주시기 바랍니다.
