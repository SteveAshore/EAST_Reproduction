
# put your training data as TRAINING_DATA_PATH. Entropy exponent and coefficient are calculated based on the training data.
EXPONENT=1
COEFF=6.8
GPU=0
TRAINING_DATA_PATH='Datasets/processed/1b_gsm8k_pair.json'

CUDA_VISIBLE_DEVICES=$GPU ACCELERATE_LOG_LEVEL=info accelerate launch --config_file gpu.yaml dpo.py \
    --dataset_name $TRAINING_DATA_PATH \
    --model_name_or_path Meta-Llama-3.2-1B-Instruct \
    --learning_rate 2e-7 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --output_dir ./model/Llama-3.2-1B-NEW-dpo-gsm8k-3epoch-lr2e-7-bs16-beta0.01-entropy_non_linear1 \
    --beta 0.01 \
    --no_remove_unused_columns \
    --report_to wandb \
    --save_strategy "no" \
    --warmup_ratio 0.1 \
    --bf16 True