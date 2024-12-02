from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
login(token="hf_token")


def prepare_datasets(args):
    datasets = load_dataset(args.data_path)
    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]
    test_dataset = datasets["test"]

    # 증강한 데이터가 있다면 아래 코드 실행
    # augmented_train_dataset = load_dataset("json" ,data_files="augmented_train_dataset.jsonl")["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    special_tokens_dict = {
        'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&',
                                      '&bank-account&', '&num&', '&online-account&']
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    def tokenize_function(dataset):
        return tokenizer(
            dataset["input"],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False
        )

    return (
        train_dataset.map(tokenize_function, batched=True).rename_column("output", "labels"),
        # 증강한 데이터가 있다면 아래 코드 실행
        # augmented_train_dataset.map(tokenize_function, batched=True).rename_column("output", "labels"),
        val_dataset.map(tokenize_function, batched=True).rename_column("output", "labels"),
        test_dataset.map(tokenize_function, batched=True).rename_column("output", "labels")
    )
