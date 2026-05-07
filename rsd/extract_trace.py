import json

import os
import shutil
import zipfile
from glob import glob
import ast
import pickle

files = glob("../data/raw_logs/tau2bench/airline/gpt-4.1/expert/*.json")
files = sorted(files)

save_dir = "../traces/tau2bench/gpt-4.1_w_log_probs/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

reward_dict = {}

for i, file in enumerate(files):
    with open(file, "r") as f:
        logs = json.load(f)

    samples = logs['samples']

    for idx, sample in enumerate(samples):
        traj = json.loads(sample['output']['choices'][0]['message']['content'])['trajectories']
        reward_arr = ast.literal_eval(sample['scores']['model_graded_fact_subgoals_auc']['value']['progress_turns'])

        progress_rate = reward_arr[-1]
        
        reward_dict[f"trajectories_{i}_{idx}"] = progress_rate

        with open(f"{save_dir}/trajectories_{i}_{idx}.json", "w") as out_f:
            json.dump(traj, out_f, indent=2)
            
        with open(f"{save_dir}/reward_dict.pkl", "wb") as f:
            pickle.dump(reward_dict, f)