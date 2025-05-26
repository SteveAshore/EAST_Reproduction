import json
import numpy as np
from collections import Counter
import random
import math
random.seed(42)

system_prompt="Please reason step by step, and put your final answer within \\boxed{{}}."
manual_discard=[]
# manual_discard=[1,75,23,44,7469,7402,7359,7343,7333,7306,7361,7251,7239,7231,7220,7208,7174,7119,7088,6609,6664,6687,6777,6809,6817,6829,6848,6860,6879,6944,6963,6988,7013,7010,7031,6537,6468,6466,6400,6360,6358,6328,6260,6180,6164,6089,6049,6032,5986,5938,5921,5867,5845,5823,5821,5771,5766,5750,5743,5708,5707,5704,5682,5661,5636,5592,5559,5543,5541,5499,5473,5437,5372,5358,5343,5288,5283,5245,5168,5157,5152,5108,4788,4803,4813,4828,4819,4839,4883,4903,4949,4960,4982,5000,5010,5011,5029,5088,4767,4756,4712,4678,4669,4632,4629,4622,4617,4605,4478,4470,4469,4439,4436,4418,4390,4366,4352,4351,4344,4315,4297,4294,4293,4263,4252,4243,4238,4224,4212,4116,4110,4088,4076,4071,4067,4017,4012,3984,3960,3958,3954,3937,3906,3884,3873,3841,3844,3776,3774,3743,3724,3721,3720,3706,3695,3686,3679,3640,3621,3619,3611,3599,3582,3530,3527,3519,3510,3007,3046,3051,3052,3059,3068,3075,3093,3147,3206,3219,3261,3265,3312,3366,3393,3406,3409,3438,3463,2986,2982,2916,2885,2876,2846,2791,2708,2661,2653,2649,2646,2615,2585,2567,1414,1371,102,93,91,2544,2531,2518,2501,2498,2492,2461,2420,2363,2202,2192,2196,2181,2170,2145,1559,1521,1495,1478,1467,1458,2130,2127,2102,2088,1986,1963,1950,1938,1917,1915,1858,1846,1725,1723,1712,1666,1657,1645,1625,1590,1578,1348,1320,1306,1293,1135,1083,1075,1064,999,971,941,926,903,853,826,800,760,754,721,701,682,673,651,619,613,585,544,478,433,423,379,368,283,270,280,258,250,223,153,110]
def get_distribution(string_list):
    counter = Counter(string_list)
    topk = counter.most_common(1)
    return topk[0]

def calculate_probabilities(numbers):
    """
    Calculate the probabilities of each number based on its frequency in the given list.

    Parameters:
        numbers (list of int/float): A list of numbers.

    Returns:
        list of float: A list of probabilities where each probability corresponds to the popularity
                       of the respective number in the input list.
    """
    # Count the occurrences of each number
    counts = Counter(numbers)
    total_count = len(numbers)
    
    # Calculate the probability for each number in the list
    seen = set()
    probabilities = [counts[num] / total_count for num in numbers if num not in seen and not seen.add(num)]
    
    return probabilities

def calculate_entropy(probabilities, base=2, normalized=False):
    """
    Calculate the entropy given a list of probabilities.

    Parameters:
        probabilities (list of float): A list of probabilities for each event. 
                                       Each probability should be in the range [0, 1], and 
                                       they should sum to 1.
        base (int): The base of the logarithm, default is 2 (information entropy).

    Returns:
        float: The entropy of the probability distribution.
    """
    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Probabilities must be in the range [0, 1].")
    
    entropy = -sum(p * math.log(p, base) for p in probabilities if p > 0)
    return entropy


def check_box(resp):
    return '\\boxed' in resp

def construct_pair_fn(data):
    weighted_pair=[]
    count_no_box=0
    count_all_neg=0
    count_all_pos=0
    count_all_pos_nobox=0
    count_all_neg_nobox=0
    count_score_wrong_before=0
    manual_discard_idx=0
    for instance in data:
        if instance['idx'] in manual_discard:
            manual_discard_idx+=1
            continue
        if instance['gt'] not in instance['pred']:
            count_all_neg+=1
            continue 

        prediction=instance['pred']
        curr_n = len(prediction) 
        n = len(instance['pred'])
        correct_answers = [instance['code'][i] for i in range(n) if instance['score'][i] and instance["pred"][i]==instance['gt']]

        prev_correct_n=instance['score'].count(True)
        if instance['pred'].count(instance['gt']) > prev_correct_n:
            count_score_wrong_before+=1
            continue

        if len(correct_answers)==n:
            count_all_pos+=1
            continue
        correct_answers = [resp for resp in correct_answers if check_box(resp)]
        if len(correct_answers)==0:
            count_all_pos_nobox+=1
            continue

        incorrect_answers_string=[instance['pred'][i] for i in range(n) if instance['score'][i]==False and check_box(instance['code'][i])]
        # make sure the incorrect answer is not boxed
        if len(incorrect_answers_string)==0:
            count_all_neg_nobox+=1
            continue

        common_answer=get_distribution(incorrect_answers_string)[0]
        incorrect_popular_answers=[instance['code'][i] for i in range(n) if common_answer == instance['pred'][i]]
        correct_answer=random.choice(correct_answers)
        correct_answer=[{"role": "system", "content":  system_prompt}, {"role": "user", "content":  instance['question']}, {"role": "assistant", "content":correct_answer }]
        chosen_weight=instance['score'].count(True)/curr_n
        
        incorrect_popular_answer=random.choice(incorrect_popular_answers)
        
        rejected_weight=instance['pred'].count(common_answer)/curr_n
        incorrect_popular_answer=[{"role": "system", "content": system_prompt}, {"role": "user", "content":  instance['question']}, {"role": "assistant", "content":incorrect_popular_answer }]
        prob=calculate_probabilities(prediction)
        entropy=calculate_entropy(prob)
    
        weighted_pair.append({'prompt':instance['question'], 'chosen':correct_answer,'rejected':incorrect_popular_answer, 
                              'chosen_weight':chosen_weight, 'rejected_weight':rejected_weight, 'entropy':entropy, "boxed_n": curr_n})
    print(count_no_box)
    print(count_all_neg)
    print(count_all_pos_nobox)
    print(count_all_neg_nobox)
    print(count_all_pos)
    print(count_score_wrong_before)
    print(manual_discard_idx)
    return weighted_pair

def get_data_distribution(data, metric):
    chosen_weight=[]
    rejected_weight=[]
    entropy=[]
    for instance in data:
        chosen_weight.append(float(instance['chosen_weight']))
        rejected_weight.append(float(instance['rejected_weight']))
        entropy.append(float(instance['entropy']))
    metric['chosen_weight']=np.mean(chosen_weight)
    metric['rejected_weight']=np.mean(rejected_weight)
    metric['entropy']=np.mean(entropy)
    metric['size']=len(data)
    return metric

def metric(data,k=1):
    
    k=int(k)
    path_k=[]
    maj_k=[]
    for ele in data:
        if True in ele['correctness'][:k]:
            path_k.append(True)
        else:
            path_k.append(False)
        most_common_item = Counter(ele['pred'][:k]).most_common(1)[0][0]
        index_of_most_common = ele['pred'][:k].index(most_common_item)
        maj_k.append(True) if ele['correctness'][index_of_most_common] else maj_k.append(False)
    result_json={f'path_{k}': path_k.count(True)/len(data), f'maj_{k}':maj_k.count(True)/len(data)}
    if "type" not in data[0]:
        return result_json
    type_scores_maj = {}
    type_scores_path = {}
    for i,sample in enumerate(data):
        if sample['type'] not in type_scores_maj:
            type_scores_maj[sample['type']] = []
            type_scores_path[sample['type']] = []
        type_scores_maj[sample['type']].append(maj_k[i])
        type_scores_path[sample['type']].append(path_k[i])
    type_scores_maj = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in type_scores_maj.items()}
    type_scores_path = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in type_scores_path.items()}
    type_scores_maj = {k: v for k, v in sorted(type_scores_maj.items(), key=lambda item: item[0])}
    type_scores_path = {k: v for k, v in sorted(type_scores_path.items(), key=lambda item: item[0])}
    result_json[f'type_acc_maj_{k}'] = type_scores_maj
    result_json[f'type_acc_path_{k}'] = type_scores_path

    level_scores_maj = {}
    level_scores_path = {}
    for i,sample in enumerate(data):
        if sample['level'] not in level_scores_maj:
            level_scores_maj[sample['level']] = []
            level_scores_path[sample['level']] = []
        level_scores_maj[sample['level']].append(maj_k[i])
        level_scores_path[sample['level']].append(path_k[i])
    level_scores_maj = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in level_scores_maj.items()}
    level_scores_path = {k: np.round(np.array(v).mean() * 100, decimals=3) for k, v in level_scores_path.items()}
    level_scores_maj = {k: v for k, v in sorted(level_scores_maj.items(), key=lambda item: item[0])}
    level_scores_path = {k: v for k, v in sorted(level_scores_path.items(), key=lambda item: item[0])}
    result_json[f'level_acc_maj_{k}'] = level_scores_maj
    result_json[f'level_acc_path_{k}'] = level_scores_path  
        
    return result_json

if __name__ == "__main__":
    file=''
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    file_metric=''
    with open(file_metric, 'r') as f:
        metrics = json.load(f)
    print(type(metrics))

    weighted_pair=construct_pair_fn(data)
    # metric=metric[0]
    metrics=get_data_distribution(weighted_pair, metrics)
    with open(file.replace(".json", "_pair.json"), "w") as f:
        json.dump(weighted_pair, f, indent=4)
    with open(file.replace(".json", "_metric.json"), "w") as f:
        json.dump(metrics, f)
    print(metrics)