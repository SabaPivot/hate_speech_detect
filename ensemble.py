import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from data import prepare_datasets

def load_and_predict_model(model_dir, hate_test, device):
    """Load model and make predictions for a single model"""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    accumulated_probs = torch.zeros(len(hate_test), 2).to(device)
    
    for sample_idx, data in enumerate(tqdm(hate_test)):
        input_ids = torch.tensor([data['input_ids']]).to(device)
        attention_mask = torch.tensor([data['attention_mask']]).to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions = F.softmax(logits, dim=-1).squeeze(0)
            accumulated_probs[sample_idx] += predictions
            
    return accumulated_probs

def get_ensemble_predictions(accumulated_probs):
    """Convert accumulated probabilities to final predictions"""
    ensemble_predictions = torch.zeros(accumulated_probs.size(0), dtype=torch.int)
    for idx in range(len(accumulated_probs)):
        ensemble_predictions[idx] = int(torch.argmax(accumulated_probs[idx], dim=0))
    return ensemble_predictions

def apply_hate_vocabulary_rules(test_dataset, predictions):
    """Apply hate vocabulary rules to adjust predictions"""
    hate_voca = [
        "치매", "극좌", "극우", "한남", "한녀", "새끼", "페미", "똘추", "부랄", "발광", 
        "벌레", "꼰대", "트랜스젠더", "트젠", "레즈", "게이", "미친놈", "느개비", "니애미", 
        "쿰척", "냄저", "재기", "창놈", "창녀", "사회악", "자살", "인셀", "여시", "지잡", 
        "씹떡", "씹덕", "또라이", "노인네", "정병", "병신", "ㅄ"
    ]
    
    result_array = np.array(predictions)
    count = 0
    
    for i, data in enumerate(test_dataset):
        if any(word in data["input"] for word in hate_voca) and result_array[i] != 1:
            count += 1
            print(data["input"])
            result_array[i] = 1
            
    print(f"Adjusted {count} predictions based on hate vocabulary")
    return result_array

def softvote(args):
    """Main ensemble voting function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accumulated_probs = None
    
    for model_dir, model_name in zip(args.model_dir, args.model_name):
        print(f"Processing model: {model_name} from {model_dir}")
        args.model_dir = model_dir
        args.model_name = model_name
        
        _, _, hate_test = prepare_datasets(args)
        
        model_probs = load_and_predict_model(model_dir, hate_test, device)
        if accumulated_probs is None:
            accumulated_probs = model_probs
        else:
            accumulated_probs += model_probs
    
    ensemble_predictions = get_ensemble_predictions(accumulated_probs)
    
    test_dataset = load_dataset(args.data_path)["test"]
    if 'output' in test_dataset.column_names:
        test_dataset = test_dataset.remove_columns('output')
    
    # Rule_based filter 적용
    final_predictions = apply_hate_vocabulary_rules(test_dataset, ensemble_predictions)
    
    # 결과 저장
    test_dataset = test_dataset.add_column("output", final_predictions.tolist())
    test_dataset.to_json(args.jsonl_path, orient='records', lines=True, force_ascii=False)
    print("Evaluation completed successfully!")
