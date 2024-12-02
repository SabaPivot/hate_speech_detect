import wandb
import argparse
from model import load_model_and_inference
from model import train
from ensemble import softvote

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="SabaPivot/hate_speech"
    )
    parser.add_argument(
        "--model_name", type=str, nargs="+", default="xlm-roberta-base"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        help='모델 타입 (예: "bert", "electra")',
    )
    parser.add_argument(
        "--save_path", type=str, default="./model", help="모델 저장 경로"
    )
    parser.add_argument(
        "--save_step", type=int, default=200, help="모델을 저장할 스텝 간격"
    )
    parser.add_argument(
        "--logging_step", type=int, default=200, help="로그를 출력할 스텝 간격"
    )
    parser.add_argument(
        "--eval_step", type=int, default=200, help="모델을 평가할 스텝 간격"
    )
    parser.add_argument(
        "--save_limit", type=int, default=5, help="저장할 모델의 최대 개수"
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 값")
    parser.add_argument("--epochs", type=int, default=5, help="에폭 수")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 사이즈",
    )
    parser.add_argument(
        "--max_len", type=int, default=256, help="입력 시퀀스의 최대 길이"
    )
    parser.add_argument("--lr", type=float, nargs="+", default=3e-5, help="학습률(learning rate)")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="가중치 감소(weight decay) 값"
    )
    parser.add_argument("--warmup_steps", type=int, default=300, help="워밍업 스텝 수")
    parser.add_argument(
        "--early_stopping_patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--scheduler", type=str, default="linear", help="학습률 스케줄러 타입"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        nargs="+",
        default="./best_model",
        help="추론 시 불러올 모델의 경로",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="bert-test",
        help="wandb 에 기록되는 run name",
    )

    parser.add_argument(
        "--data_type",
        type=str,
        default="original",
        help="데이터 증강 시, 데이터 총 갯수를 입력"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="do_train",
        help="모델을 훈련시킬지 Evaluation 할지"
    )

    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="./jsonl_test_result.jsonl",
        help="Eval시 결과 저장 경로"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    learning_rates = args.lr
    if args.mode == "do_train":
        # Iterate over learning rates
        for args.lr in learning_rates:
            args.run_name = f"Augmented_{args.lr}_32_50_12_classifier_dropout=0.2_fp16_klue/bert"
            args.model_dir = f"/home/careforme.dropout/huggingface_trainer/Augmented_{args.lr}_32_50_12_classifier_dropout=0.1_fp16_monologg/kobert"
            wandb.init(
                project="hate_speech",
                name=args.run_name,
                config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "data": args.data_type
           })
            train(args)
            wandb.finish()

    elif args.mode == "do_eval":
        load_model_and_inference(args)

    elif args.mode == "do_ensemble":
        softvote(args)
