import pytorch_lightning as pl
import torch
import os
from data import prepare_datasets
from tqdm import tqdm
from utils import compute_metrics
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import load_dataset

def load_model(args):
    model_name = args.model_name
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = 2
    print(model_config)

    model_config.classifier_dropout = 0.1
    print(model_config)

    # model 설정해주기
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    tokenizer_len = len(AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)) + 8
    model.resize_token_embeddings(tokenizer_len)
    print(model)

    return model

def load_trainer_for_train(args, model, hate_train_dataset, hate_valid_dataset):
    training_args = TrainingArguments(
        output_dir=args.model_dir,  # output directory
        save_total_limit=args.save_limit,  # number of total save model.
        save_steps=args.save_step,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.save_path + "/logs",  # directory for storing logs
        logging_strategy="epoch",
        eval_strategy="epoch",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=args.eval_step,  # evaluation step.
        load_best_model_at_end=True, # 마지막에 Save하는 모델은 무조건 가장 좋은 모델
        save_strategy="epoch",
        report_to="wandb",  # W&B 로깅 활성화
        run_name=args.run_name,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        gradient_accumulation_steps=10,
    )

    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=0.001
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay, amsgrad=False, betas=(0.9, 0.999)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hate_train_dataset,
        eval_dataset=hate_valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MyCallback],
        optimizers=(
            optimizer,
            get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps,
                num_training_steps=(len(hate_train_dataset) // args.batch_size) * args.epochs // 10
            )
        )
    )
    print("---Set Trainer Done---")

    return trainer

def train(args):
    pl.seed_everything(seed=42, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args)
    model.to(device)
    train_dataset, valid_dataset, test_dataset = prepare_datasets(args)

    trainer = load_trainer_for_train(args, model, train_dataset, valid_dataset)

    print("---Start Training---")
    trainer.train()
    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"The best model is saved at: {best_checkpoint}")
    print("---End Training---")
    model.save_pretrained(args.model_dir)


def load_model_and_inference(args):
    for model_dir, model_name in zip(args.model_dir, args.model_name):
        args.model_dir = model_dir
        args.model_name = model_name
        print(model_dir, model_name)

        _, _, hate_test = prepare_datasets(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        model.to(device)
        model.eval()

        test_result = []
        for data in tqdm(hate_test):
            input_ids, attention_mask = torch.tensor([data['input_ids']]).to(device), torch.tensor([data['attention_mask']]).to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                predictions = torch.argmax(logits, dim=-1)
                test_result.extend([int(predictions[0])])

        test_dataset = load_dataset(args.data_path)["test"]

        # Remove existing 'output' column if it exists
        if 'output' in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns('output')

        hate_voca = ["치매", "극좌", "극우", "한남", "한녀", "새끼", "페미", "똘추", "부랄", "발광", "벌레", "꼰대", "트랜스젠더", "트젠", "레즈", "게이",
                     "미친놈", "느개비", "니애미", "쿰척", "냄저", "재기", "창놈", "창녀", "사회악", "자살", "인셀", "여시", "지잡", "씹떡", "씹덕",
                     "또라이", "노인네", "정병", "병신", "ㅄ"]

        count = 0
        for i in range(len(test_dataset)):  # Use index-based iteration
            data = test_dataset[i]  # Access the original entry using the index
            if any(word in data["input"] for word in hate_voca):
                if test_result[i] != 1:
                    count += 1
                    print(data["input"])
                    test_result[i] = 1  # Update the output directly in the dataset
        print(count)

        # Add the new predictions to the dataset
        test_dataset = test_dataset.add_column("output", test_result)
        args.jsonl_path = f"{args.model_name}_records.jsonl"
        test_dataset.to_json(args.jsonl_path, orient='records', lines=True, force_ascii=False)
        print("Evaluation done!")

