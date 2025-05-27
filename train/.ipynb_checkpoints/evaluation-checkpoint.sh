# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen-boxed"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="model/Llama-3.2-1B-NEW-dpo-gsm8k-3epoch-lr2e-7-bs16-beta0.01-entropy_non_linear1"
bash sh/eval_gsm8k_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH