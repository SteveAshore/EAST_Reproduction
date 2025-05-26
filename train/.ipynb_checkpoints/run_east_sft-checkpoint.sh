
python -c "import torch; torch.cuda.empty_cache()"

# put your training data as TRAINING_DATA_PATH. Entropy exponent and coefficient are calculated based on the training data.
EXPONENT=1
COEFF=6.8

export PYTORCH_SHOW_CPP_STACKTRACES=1
export ACCELERATE_DEBUG_MODE="1"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file gpu_sft.yaml sft.py \
    --dataset_name Datasets/processed/1b_gsm8k_pair.json \
    --model_name_or_path Meta-Llama-3.2-1B-Instruct \
    --learning_rate 2e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --output_dir ./model/Llama-3.2-1B-NEW-sft-gsm8k-3epoch-lr2e-6-bs16-entropy_non_linear1 \
    --no_remove_unused_columns \
    --report_to wandb \
    --save_strategy "no" \
    --warmup_ratio 0.1 \
    --use_weighting \
    --use_entropy_non_linear $EXPONENT \
    --use_entropy_non_linear_coeff $COEFF \
    # --bf16 True