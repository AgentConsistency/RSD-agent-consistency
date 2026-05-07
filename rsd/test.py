import sys
from train import Config, MICONetwork, TraceLoader, set_seed, TracePreprocessor, RSDDataset
from glob import glob
import random
import pickle
import json
import os

import torch

def load_model(config):
    model = MICONetwork(config)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    return model

if __name__ == "__main__":
    config = Config()
    config.embedding_cache_path = config.embedding_cache_path.replace("cached_transitions.pt", "cached_transitions_test.pt")
    model = load_model(config)
    
    results = []
    
    set_seed(config.seed)
    
    print(f"--- Configuration ---")
    print(f"Latent Dim: {config.latent_dim}")
    print(f"Beta: {config.beta}")    
    print(f"Trajectories Path: {config.trajectories_path}")
    loader = TraceLoader()
    
    # files = glob(config.trajectories_path + "*.json")
    
    # files.sort()
    
    # # Randomly select 80% for training, 20% for testing
    # random.shuffle(files)
    # train_size = int(0.8 * len(files))
    # test_files = files[train_size:]
    
    files = glob(config.trajectories_path + "*.json")
    # random.shuffle(files)
    # test_files = files
    files.sort()
    
    # Extract unique sample_ids from filenames (pattern: trajectories_<run_id>_<sample_id>.json)
    sample_ids = set()
    for f in files:
        filename = os.path.basename(f)
        # Extract sample_id from pattern trajectories_<run_id>_<sample_id>.json
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            sample_id = parts[2]  # Third part is sample_id
            sample_ids.add(sample_id)
    
    sample_ids = sorted(list(sample_ids))
    
    # Randomly select 80% of unique sample_ids for training
    random.shuffle(sample_ids)
    train_size = int(0.8 * len(sample_ids))
    train_sample_ids = set(sample_ids[:train_size])
    test_sample_ids = set(sample_ids[train_size:])
    
    print(f"Total unique samples: {len(sample_ids)}")
    print(f"Training samples: {len(train_sample_ids)}")
    print(f"Test samples: {len(test_sample_ids)}")
    
    # Filter files to only include those with sample_ids in training set (all runs for those samples)
    test_files = []
    for f in files:
        filename = os.path.basename(f)
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            sample_id = parts[2]
            if sample_id in test_sample_ids:
                test_files.append(f)
    
    print(f"Total test files (all runs for test samples): {len(test_files)}")
    
    reward_dict_path = config.trajectories_path + "reward_dict.pkl"
    progress_rates = pickle.load(open(reward_dict_path, "rb"))
    
    # Load all test transitions at once
    all_test_transitions = []
    file_transition_ranges = []  # Store (start_idx, end_idx, file_path) for each file
    
    for trace in test_files:
        transitions = loader.load_file(trace)
        if len(transitions) == 0:
            continue
        start_idx = len(all_test_transitions)
        all_test_transitions.extend(transitions)
        end_idx = len(all_test_transitions)
        file_transition_ranges.append((start_idx, end_idx, trace))
    
    print(f"Total test transitions: {len(all_test_transitions)}")
    
    # Process all transitions at once (this will use the cache properly)
    preprocessor = TracePreprocessor(config)
    processed_data = preprocessor.process(all_test_transitions)
    
    dataset = RSDDataset(processed_data)
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Compute scores for all items
    all_scores = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            z_current = item["z_current"]
            phi_start = model(z_current.unsqueeze(0))
            u_start_start = model.compute_distance(phi_start, phi_start)
            all_scores.append(u_start_start.item())
    
    # Group scores by original file
    for start_idx, end_idx, trace in file_transition_ranges:
        trace_result = {
            "trace_path": trace,
            "scores": all_scores[start_idx:end_idx]
        }
        if len(all_scores[start_idx:end_idx]) == 1:
            continue
        
        trace_result["progress_rate"] = progress_rates[trace.split("/")[-1].replace(".json", "")]
        results.append(trace_result)
    
    # output_file = f"results/{config.embedding_cache_path.split('/')[-2]}/{config.save_path.split('/')[-2].replace('checkpoints_', '')}/distance_scores.json"
    output_file = config.distance_scores_path

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved results to {output_file}")