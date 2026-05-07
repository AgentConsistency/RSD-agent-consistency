from dataclasses import dataclass
import json
import os
import random
from typing import List, Tuple
from tqdm import tqdm
from glob import glob
import time

import backoff
import requests
from openai import InternalServerError, RateLimitError

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    input_dim: int = 1536
    latent_dim: int = 256
    mico_start_dim: int = 512
    beta: float = 5.0 # angle weight
    alpha: float = 0.5 # norm weight 
    agent_model: str = "gpt-4.1"
    dataset_name: str = "tau2bench"
    embedding_cache_path: str = f"./embeddings/{dataset_name}/{agent_model}/cached_transitions.pt"
    force_recompute: bool = False
    mode: str = "train"
    
    embeddings_batch_size: int = 64
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.8
    tau: float = 0.005
    n_steps: int = 3
    epochs: int = 1000
    model_name: str = 'embed-v-4-0'
    seed: int = 42
    
    save_path: str = f"models/{dataset_name}/{agent_model}/checkpoints_gamma={gamma}_beta={beta}_dim={latent_dim}_n_steps={n_steps}/rsd_final.pth"
    
    trajectories_path: str = f"traces/{dataset_name}/{agent_model}/"

    distance_scores_path: str = f"results/{dataset_name}/{agent_model}/{save_path.split('/')[-2].replace('checkpoints_', '')}/distance_scores.json"

    
class MICONetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.beta = config.beta
        self.alpha = config.alpha
        self.phi = nn.Sequential(
            nn.Linear(config.input_dim, config.mico_start_dim),
            nn.LayerNorm(config.mico_start_dim),
            nn.ReLU(),
            nn.Linear(config.mico_start_dim, config.mico_start_dim // 2),
            nn.ReLU(),
            nn.Linear(config.mico_start_dim // 2, config.latent_dim)
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        return self.phi(x)
    
    def compute_distance(self, phi_x, phi_y):
        norm_x_sq = torch.mean(phi_x**2, dim=1, keepdim=True)
        norm_y_sq = torch.mean(phi_y**2, dim=1, keepdim=True)
        
        phi_x_n = F.normalize(phi_x, p=2, dim=1)
        phi_y_n = F.normalize(phi_y, p=2, dim=1)
        
        cos_sim = torch.sum(phi_x_n * phi_y_n, dim=1, keepdim=True)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        
        theta = torch.acos(cos_sim)
        
        return self.alpha * (norm_x_sq + norm_y_sq) + self.beta * theta  
    
class TraceLoader:
    def linearize_json(self, data):
        if isinstance(data, dict):
            return ", ".join([f"{k} is {self.linearize_json(v)}" for k, v in data.items() if v is not None and k != "reasoning"])
        elif isinstance(data, list):
            return "; ".join([self.linearize_json(item) for item in data])
        else:
            return str(data)
        
    def format_turn(self, turn):
        role = turn["role"]
        content = turn.get("content")
        if role == 'user':
            return f"User: {content}"
        elif role == 'tool':
            try:
                tool_data = json.loads(content)
                readable_data = self.linearize_json(tool_data)
                return f"System: {readable_data}"
            except Exception as e:
                return f"System: {content}"
        elif role == 'agent':
            if turn.get('tool_calls'):
                calls = []
                for tc in turn['tool_calls']:
                    func_name = tc['name']
                    func_args = self.linearize_json(tc['arguments'])
                    calls.append(f"calls tool '{func_name}' with arguments {func_args}")
                return f"Agent: {', '.join(calls)}"
            else:
                return f"Agent: {content}"
            
    def load_file(self, file_path):
        with open(file_path, 'r') as f:
            raw_traces = json.load(f)
        transitions = []
        history_buffer = []
        for conversation in raw_traces:
            i = 0
            while i < len(conversation):
                turn = conversation[i]
                turn_text = self.format_turn(turn)
                if turn['role'] == 'agent':
                    state_str = " [SEP] ".join(history_buffer)
                    if not state_str: state_str = "Start of conversation"
                    output_str = turn_text
                    next_history = history_buffer + [turn_text]
                    j = i + 1
                    while j < len(conversation):
                        next_turn = conversation[j]
                        next_turn_text = self.format_turn(next_turn)
                        if next_turn['role'] == 'tool':
                            next_history.append(next_turn_text)
                            j += 1
                        elif next_turn['role'] == 'user':
                            next_history.append(next_turn_text)
                            break
                        else:
                            break
                    next_state_str = " [SEP] ".join(next_history)
                    if j >= len(conversation):
                        done = True
                    else:
                        done = False
                    transitions.append((state_str, output_str, next_state_str, done))
                    history_buffer = next_history.copy()
                    i = j
                else:
                    history_buffer.append(turn_text)
                    i += 1
        return transitions
            
class TracePreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self._load_client()
        
    def _load_client(self):
        if self.client is None:
            from cohere_embed_v4 import get_client
            self.client = get_client()
            
    def _get_embeddings(self, texts: List[str]):
        batch_size = self.config.embeddings_batch_size
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
        return torch.tensor(all_embeddings, dtype=torch.float32)
    
    @backoff.on_exception(
        backoff.constant,
        (RateLimitError, InternalServerError, requests.exceptions.Timeout, requests.exceptions.ConnectionError),
        interval=60,
        max_tries=25,
        jitter=None,
        on_backoff=lambda details: print(f"Backing off {details['wait']}s after {details['tries']} tries...")
    )
    def _get_batch_embeddings(self, batch_texts: List[str]):
        response = self.client.embeddings.create(
            input=batch_texts,
            model=self.config.model_name,
            dimensions=self.config.input_dim,
            encoding_format="float",
            extra_body={"truncate": "START"},
            extra_headers={"extra-parameters": "pass-through"}
        )
        return [item.embedding for item in response.data]        
            
            
    def process(self, transitions: List[Tuple[str, str, str, bool]]):
        num_samples = len(transitions)
        next_indices = torch.full((num_samples,), -1, dtype=torch.long)
        for i in range(num_samples - 1):
            _, _, _, done = transitions[i]
            if not done:
                next_indices[i] = i + 1
            else:
                next_indices[i] = -1
                
        if os.path.exists(self.config.embedding_cache_path) and not self.config.force_recompute:
            print(f"Loading cached embeddings from {self.config.embedding_cache_path}")
            try:
                data = torch.load(self.config.embedding_cache_path)
                if len(data["outputs"]) == len(transitions):
                    data["done"] = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
                    data["next_indices"] = next_indices
                    return data
                else:
                    print("Cached embeddings size mismatch, recomputing embeddings.")
            except Exception as e:
                print(f"Error loading cached embeddings: {e}. Recomputing embeddings.")
                
        print("Computing embeddings for transitions...")
        states = [t[0] for t in transitions]
        outputs = [t[1] for t in transitions]
        next_states = [t[2] for t in transitions]
        z_current = self._get_embeddings(states)
        z_next = self._get_embeddings(next_states)
        z_outputs = self._get_embeddings(outputs)
        
        data = {
            "z_current": z_current,
            "z_next": z_next,
            "z_outputs": z_outputs,
            "outputs": outputs,
            "done": torch.tensor([t[3] for t in transitions], dtype=torch.float32),
            "next_indices": next_indices
        }
        
        print(f"Saving embeddings to {self.config.embedding_cache_path}")
        
        if not os.path.exists(os.path.dirname(self.config.embedding_cache_path)):
            os.makedirs(os.path.dirname(self.config.embedding_cache_path))
            
        torch.save(data, self.config.embedding_cache_path)
        return data
    
class RSDDataset(Dataset):
    def __init__(self, data_dict):
        self.z_current = data_dict["z_current"]
        self.z_next = data_dict["z_next"]
        self.z_outputs = data_dict["z_outputs"]
        self.outputs = data_dict["outputs"]
        self.done = data_dict["done"]
        self.next_indices = data_dict["next_indices"]
        
    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, idx):
        return {
            "index": idx,
            "z_current": self.z_current[idx],
            "z_next": self.z_next[idx],
            "z_outputs": self.z_outputs[idx],
            "output_text": self.outputs[idx],
            "done": self.done[idx],
        }
        
class RSDTrainer:
    def __init__(self, config: Config, model: MICONetwork, dataset: RSDDataset):
        self.config = config
        self.model = model
        self.target_model = MICONetwork(config)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        self.full_z_outputs = dataset.z_outputs
        self.full_z_current = dataset.z_current
        self.full_z_next = dataset.z_next
        self.full_dones = dataset.done
        self.full_next_indices = dataset.next_indices
        
        self.n_steps = self.config.n_steps
        
    def soft_update(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
            )
            
    def get_proxy_reward(self, z_out_a: torch.Tensor, z_out_b: torch.Tensor):
        sim_score = F.cosine_similarity(z_out_a, z_out_b, dim=1, eps=1e-8)
        dist = 1.0 - sim_score
        scale = 20.0
        shift = 0.15
        r_proxy = torch.sigmoid(scale * (dist - shift))
        
        return r_proxy.unsqueeze(1)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        steps = 0
        
        for batch_a in tqdm(dataloader):
            idx_a = batch_a["index"]
            roll_idx = random.randint(1, len(idx_a) - 1)
            idx_b = torch.roll(idx_a, shifts=roll_idx, dims=0)
            
            with torch.no_grad():
                accumulated_reward = 0
                current_gamma = 1.0
                
                curr_a = idx_a.clone()
                curr_b = idx_b.clone()
                
                active_mask = torch.ones_like(curr_a, dtype=torch.float32)
                
                for _ in range(self.n_steps):
                    z_out_a = self.full_z_outputs[curr_a]
                    z_out_b = self.full_z_outputs[curr_b]
                    
                    r_proxy = self.get_proxy_reward(z_out_a, z_out_b).squeeze(1)
                    accumulated_reward += current_gamma * r_proxy * active_mask
                    
                    current_gamma *= self.config.gamma
                    
                    next_a = self.full_next_indices[curr_a]
                    next_b = self.full_next_indices[curr_b]
                    
                    done_a = (next_a == -1).float()
                    done_b = (next_b == -1).float()
                    
                    step_mask = (1.0 - done_a) * (1.0 - done_b)
                    active_mask = active_mask * step_mask
                    
                    curr_a = torch.where(next_a != -1, next_a, curr_a)
                    curr_b = torch.where(next_b != -1, next_b, curr_b)
                
                phi_next_a = self.target_model(self.full_z_current[curr_a])
                phi_next_b = self.target_model(self.full_z_current[curr_b])

                d_future = self.target_model.compute_distance(phi_next_a, phi_next_b).squeeze(1)
                target = accumulated_reward + (current_gamma * active_mask * d_future)
                target = target.unsqueeze(1)
                
            phi_a = self.model(batch_a["z_current"])
            phi_b = self.model(batch_a["z_current"])[torch.arange(len(idx_a)).roll(roll_idx)]
            d_pred = self.model.compute_distance(phi_a, phi_b)
            loss = self.loss_fn(d_pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.soft_update()
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps if steps > 0 else 0.0
        self.scheduler.step(avg_loss)
        return avg_loss
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)
    
    print(f"--- Configuration ---")
    print(f"Latent Dim: {config.latent_dim}")
    print(f"Beta: {config.beta}")    
    
    loader = TraceLoader()
    
    files = glob(config.trajectories_path + "*.json")
    files.sort()
    # random.shuffle(files)
    # train_files = files
    
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
    train_files = []
    for f in files:
        filename = os.path.basename(f)
        parts = filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            sample_id = parts[2]
            if sample_id in train_sample_ids:
                train_files.append(f)
    
    print(f"Total training files (all runs for training samples): {len(train_files)}")
    
    all_transitions = []
    for f_path in train_files:
        if os.path.exists(f_path):
            all_transitions.extend(loader.load_file(f_path))
        else:
            print(f"File {f_path} does not exist, skipping.")
            
    if not all_transitions:
        print("No transitions found. Exiting.")
        exit(1)
        
    print(f"Total transitions loaded: {len(all_transitions)}")
    
    preprocessor = TracePreprocessor(config)
    processed_data = preprocessor.process(all_transitions)
    
    dataset = RSDDataset(processed_data)
    indices = list(range(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    model = MICONetwork(config)
    trainer = RSDTrainer(config, model, dataset)
    
    print("Starting training")
    for epoch in range(config.epochs):
        start_time = time.time()
        avg_loss = trainer.train_epoch(dataloader)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.6f}, Time: {duration:.2f}s")
    
    if not os.path.exists(os.path.dirname(config.save_path)):
        os.makedirs(os.path.dirname(config.save_path))
        
    trainer.save_model(config.save_path)
            
                    