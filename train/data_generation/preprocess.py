import json
import random
import numpy as np
from util import *
import json

# 1. Use data generation pipeline to generate data (https://github.com/QwenLM/Qwen2.5-Math)
# 2. Use this script to preprocess the data, you will have
#  a pair file, with each question having a chosen(positive) and rejected(negative) response
#  a metrics file, which contains the metrics of the data 
#  a extended metrics file, which contains the metrics of the data with different weight coefficients

#tips: for pair file, I did some cleaning to make sure both chosen and rejected pairs contain correct format("box{}"). there are some cleaning checks in the code and you may add more to ensure the resulting training data quality. 
#tips: for pair file, you can do a manual check to see if negative answer is incorrect (sometimes it contain correct answer but the evaluation pipeline is not able to detect it, which may affect the DPO training)
file0='../Datasets/1b_math_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
file1='../Datasets/8b_math_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
file2='../Datasets/1B_gsm8k_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
file3='../Datasets/8b_gsm8k_train_qwen25-math-cot_-1_seed0_t0.7_s0_e-1.jsonl'
#your file generated using data generation pipeline, could be a single jsonl file or multiple jsonl files
output_dir='../Datasets/processed/1b_gsm8k_new.json'

files = [file0]
new_data = []
key=['code', 'score', 'report', 'pred']
# Read and combine responses from all files
for file_path in files:
    with open(file_path, 'r') as f:
        data = []
        lines = f.readlines()  # 读取所有行
        if len(lines) > 1:  # 如果文件有多行，则跳过最后一行
            lines = lines[:-1]
        for line in lines:
            data.append(json.loads(line.strip()))
        if len(new_data) == 0:
            new_data=data
            continue
        for i, ele in enumerate(data):
            for k in key:
                new_data[i][k].extend(ele[k])
metrics={}
new_data_pair=construct_pair_fn(new_data)
metrics=get_data_distribution(new_data_pair, metrics)

with open(output_dir.replace(".json", "_pair.json"), "w") as f:
    json.dump(new_data_pair, f, indent=4)
with open(output_dir.replace(".json", "_metric.json"), "w") as f:
    json.dump(metrics, f)



a_e={-3:[],-2.5:[],-2:[],-1.5:[],-1.25:[],-1:[],-0.75:[],-0.5:[],0.1:[],0.2:[],0.5:[],0.7:[],1:[],1.5:[],1.75:[],2:[],2.5:[],3:[]}
a_a_rev={-3:[],-2.5:[],-2:[],-1.5:[],-1.25:[],-1:[],-0.75:[],-0.5:[],0.1:[],0.2:[],0.5:[],0.7:[],1:[],1.5:[],1.75:[],2:[],2.5:[],3:[]}
a_r={-3:[],-2.5:[],-2:[],-1.5:[],-1.25:[],-1:[],-0.75:[],-0.5:[],0.1:[],0.2:[],0.5:[],0.7:[],1:[],1.5:[],1.75:[],2:[],2.5:[],3:[]}

content1 = new_data_pair
content2 = metrics

chosen=[]
entropy=[]
rejected=[]
for ele in content1:
    chosen.append(ele['chosen_weight'])
    entropy.append(ele['entropy'])
    rejected.append(ele['rejected_weight'])
    for w in a_e:
        if ele['entropy']==0:
            print(ele)
        a_e[w].append(ele['entropy']**w)
        a_r[w].append(ele['rejected_weight']**w)
        a_a_rev[w].append((1-ele['chosen_weight'])**w)
print(max(chosen), min(chosen))
print(max(entropy), min(entropy), np.mean(entropy))
content2['min_entropy']=min(entropy)
content2['max_entropy']=max(entropy)
content2['min_rejected_weight']=min(rejected)
content2['max_rejected_weight']=max(rejected)

content2['min_chosen_weight']=min(chosen)
content2['max_chosen_weight']=max(chosen)

for w in a_e:
    content2[f'min_entropy_{w}']=min(a_e[w])
    content2[f'max_entropy_{w}']=max(a_e[w])
    content2[f'mean_entropy_{w}']=np.mean(a_e[w])
    content2[f'min_chosen_weight_rev_{w}']=min(a_a_rev[w])
    content2[f'max_chosen_weight_rev_{w}']=max(a_a_rev[w])
    content2[f'mean_chosen_weight_rev_{w}']=np.mean(a_a_rev[w])
    content2[f'min_rejected_weight_{w}']=min(a_r[w])
    content2[f'max_rejected_weight_{w}']=max(a_r[w])
    content2[f'mean_rejected_weight_{w}']=np.mean(a_r[w])
with open(output_dir.replace('.jsonl', '_metric_extended.jsonl'), 'w', encoding='utf-8') as f2:
    json.dump(content2, f2, ensure_ascii=False, indent=4)
