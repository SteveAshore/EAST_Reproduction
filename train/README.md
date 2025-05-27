### Pipeline
1. 原始数据‘1B_gsm8k_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl’等共4个文件，通过'preprocess.py'中数据生成pipeline得到对应的'1b_gsm8k_pair.json'、'1b_gsm8k_metric.json'和'1b_gsm8k.json'等共12个文件
2. 使用huggingface中下载的'Meta-Llama-3.2-1B-Instruct'模型运行'run_east_dpo_test.sh'脚本，得到'Meta-Llama-3.2-1B-Instruct-NEW-dpo-math-3epoch-lr2e-7-bs16-entropy_non_linear1'模型
3. 使用'evaluation.sh'进行评估
