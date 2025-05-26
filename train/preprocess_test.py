import json
import random
import numpy as np
from util import *
import json

file0='Datasets/1b_math_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
file1='Datasets/8b_math_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
file2='Datasets/1B_gsm8k_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
file3='Datasets/8b_gsm8k_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
output_dir='Datasets/processed'

files = [file1]
new_data = []
key=['code', 'score', 'report', 'pred']

# Modified reading code with error handling
for file_path in files:
    with open(file_path, 'r') as f:
        data = []
        line_number = 0
        for line in f:
            line_number += 1
            line = line.strip()
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error in file {file_path} at line {line_number}:")
                print(f"Error message: {str(e)}")
                print(f"Problematic line (first 100 chars): {line[:100]}...")
                print(f"Line length: {len(line)} characters")
                # Print the surrounding characters where the error occurred
                error_pos = e.pos
                print(f"Context around error (char {error_pos}): ...{line[max(0, error_pos-20):error_pos+20]}...")
                raise  # Re-raise the exception after printing debug info
        
        if len(new_data) == 0:
            new_data = data
            continue
        for i, ele in enumerate(data):
            for k in key:
                new_data[i][k].extend(ele[k])

metrics = {}
new_data_pair = construct_pair_fn(new_data)
metrics = get_data_distribution(new_data_pair, metrics)

# Rest of your code remains the same...
with open(output_dir.replace(".json", "_pair.json"), "w") as f:
    json.dump(new_data_pair, f, indent=4)
with open(output_dir.replace(".json", "_metric.json"), "w") as f:
    json.dump(metrics, f)

a_e = {-3:[], -2.5:[], -2:[], -1.5:[], -1.25:[], -1:[], -0.75:[], -0.5:[], 0.1:[], 0.2:[], 0.5:[], 0.7:[], 1:[], 1.5:[], 1.75:[], 2:[], 2.5:[], 3:[]}
a_a_rev = {-3:[], -2.5:[], -2:[], -1.5:[], -1.25:[], -1:[], -0.75:[], -0.5:[], 0.1:[], 0.2:[], 0.5:[], 0.7:[], 1:[], 1.5:[], 1.75:[], 2:[], 2.5:[], 3:[]}
a_r = {-3:[], -2.5:[], -2:[], -1.5:[], -1.25:[], -1:[], -0.75:[], -0.5:[], 0.1:[], 0.2:[], 0.5:[], 0.7:[], 1:[], 1.5:[], 1.75:[], 2:[], 2.5:[], 3:[]}

content1 = new_data_pair
content2 = metrics

chosen = []
entropy = []
rejected = []
for ele in content1:
    chosen.append(ele['chosen_weight'])
    entropy.append(ele['entropy'])
    rejected.append(ele['rejected_weight'])
    for w in a_e:
        if ele['entropy'] == 0:
            print(ele)
        a_e[w].append(ele['entropy']**w)
        a_r[w].append(ele['rejected_weight']**w)
        a_a_rev[w].append((1-ele['chosen_weight'])**w)

print(max(chosen), min(chosen))
print(max(entropy), min(entropy), np.mean(entropy))
content2['min_entropy'] = min(entropy)
content2['max_entropy'] = max(entropy)
content2['min_rejected_weight'] = min(rejected)
content2['max_rejected_weight'] = max(rejected)

content2['min_chosen_weight'] = min(chosen)
content2['max_chosen_weight'] = max(chosen)

for w in a_e:
    content2[f'min_entropy_{w}'] = min(a_e[w])
    content2[f'max_entropy_{w}'] = max(a_e[w])
    content2[f'mean_entropy_{w}'] = np.mean(a_e[w])
    content2[f'min_chosen_weight_rev_{w}'] = min(a_a_rev[w])
    content2[f'max_chosen_weight_rev_{w}'] = max(a_a_rev[w])
    content2[f'mean_chosen_weight_rev_{w}'] = np.mean(a_a_rev[w])
    content2[f'min_rejected_weight_{w}'] = min(a_r[w])
    content2[f'max_rejected_weight_{w}'] = max(a_r[w])
    content2[f'mean_rejected_weight_{w}'] = np.mean(a_r[w])

with open(output_dir.replace('.jsonl', '_metric_extended.jsonl'), 'w', encoding='utf-8') as f2:
    json.dump(content2, f2, ensure_ascii=False, indent=4)
