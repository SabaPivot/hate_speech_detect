import json
from glob import glob

result_files = glob(r"inference/*.jsonl")
all_predictions = []

for file in result_files:
    with open(file, 'r') as f:
        data = f.readlines()
        model_prediction = [json.loads(line) for line in data]
        all_predictions.append(model_prediction)

final_predictions = []

num_samples = len(all_predictions[0])
num_models = len(all_predictions)

for i in range(num_samples):
    votes = sum(model_predict[i]['output'] for model_predict in all_predictions)

    vote_result = 1 if votes >= (num_models / 2) else 0
    final_predictions.append({
        "id": all_predictions[0][i]['id'],
        "input": all_predictions[0][i]['input'],
        "output": vote_result
    })

with open('hard_vote.jsonl', 'w', encoding='utf-8') as f:
    for prediction in final_predictions:
        f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

print("Hard voting finished.")