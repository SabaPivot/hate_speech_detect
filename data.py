from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
login(token="hf_token")


def prepare_datasets(args):
    datasets = load_dataset(args.data_path)
    print(datasets)
    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    print(load_dataset("json" ,data_files="/home/careforme.dropout/original+smilegate.jsonl"))
    augmented_train_dataset = load_dataset("json" ,data_files="/home/careforme.dropout/original+smilegate.jsonl")["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # add special tokens
    special_tokens_dict = {
        'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&',
                                      '&bank-account&', '&num&', '&online-account&']
    }
    print(len(tokenizer))
    tokenizer.add_special_tokens(special_tokens_dict)
    print(len(tokenizer))

    # ToDo: truncation, max_length
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
        # train_dataset.map(tokenize_function, batched=True).rename_column("output", "labels"),
        augmented_train_dataset.map(tokenize_function, batched=True).rename_column("output", "labels"),
        val_dataset.map(tokenize_function, batched=True).rename_column("output", "labels"),
        test_dataset.map(tokenize_function, batched=True).rename_column("output", "labels")
    )
