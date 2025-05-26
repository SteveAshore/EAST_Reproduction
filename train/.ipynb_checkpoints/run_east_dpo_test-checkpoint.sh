#!/bin/bash

# 激活 conda 环境
# source activate seal

# 清理残留显存
python -c "import torch; torch.cuda.empty_cache()"

EXPONENT=1
COEFF=6.8

export PYTORCH_SHOW_CPP_STACKTRACES=1
export ACCELERATE_DEBUG_MODE="1"

# 1b_gsm8k_pair.json--DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file gpu.yaml dpo.py \
    --model_name_or_path Meta-Llama-3.2-1B-Instruct \
    --dataset_name Datasets/processed/1b_gsm8k_pair.json \
    --learning_rate 2e-7 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --output_dir ./model/Meta-Llama-3.2-1B-Instruct-NEW-dpo-gsm8k-3epoch-lr2e-7-bs16-entropy_non_linear1 \
    --no_remove_unused_columns \
    --report_to wandb \
    --save_strategy "no" \
    --warmup_ratio 0.1 \
    --use_entropy_non_linear $EXPONENT \
    --use_entropy_non_linear_coeff $COEFF \
    --bf16 True
