# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen-boxed"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="model/Meta-Llama-3.2-1B-Instruct-NEW-dpo-gsm8k-3epoch-lr2e-7-bs16-entropy_non_linear1"
bash sh/eval_gsm8k_math.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH