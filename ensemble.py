import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from data import prepare_datasets
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import load_dataset

def softvote(args):
    for model_dir, model_name in zip(args.model_dir, args.model_name):
        args.model_dir = model_dir
        args.model_name = model_name
        print(model_dir, model_name)

        _, _, hate_test = prepare_datasets(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = torch.zeros(len(hate_test), 2).to(device)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)
        model.to(device)
        model.eval()

        for x, data in enumerate(tqdm(hate_test)):
            input_ids, attention_mask = torch.tensor([data['input_ids']]).to(device), torch.tensor([data['attention_mask']]).to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                predictions = F.softmax(logits, dim=-1)
                predictions = predictions.squeeze(0)
                result[x] += predictions

        print(result)
        result2 = torch.zeros(result.size(0), dtype=torch.int)
        for y in range(len(result)):
            result2[y] = int(torch.argmax(result[y], dim=0))  # Use dim=0 for each row

    # Print the final result
    print(result2)

    test_dataset = load_dataset(args.data_path)["test"]

    # Remove existing 'output' column if it exists
    if 'output' in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns('output')
    # Add the new predictions to the dataset
    print(len(result2))
    print(len(test_dataset))

    result2_array = np.array(result2)

    hate_voca = ["치매", "극좌", "극우", "한남", "한녀", "새끼", "페미", "똘추", "부랄", "발광", "벌레", "꼰대", "트랜스젠더", "트젠", "레즈", "게이",
                 "미친놈", "느개비", "니애미", "쿰척", "냄저", "재기", "창놈", "창녀", "사회악", "자살", "인셀", "여시", "지잡", "씹떡", "씹덕", "또라이",
                 "노인네", "정병", "병신", "ㅄ"]

    count = 0
    for i in range(len(test_dataset)):  # Use index-based iteration
        data = test_dataset[i]  # Access the original entry using the index
        if any(word in data["input"] for word in hate_voca):
            if result2_array[i] != 1:
                count += 1
                print(data["input"])
                result2_array[i] = 1  # Update the output directly in the dataset
    print(count)
    test_dataset = test_dataset.add_column("output", result2_array.tolist())
    test_dataset.to_json(args.jsonl_path, orient='records', lines=True, force_ascii=False)
    print("Evaluation done!")
