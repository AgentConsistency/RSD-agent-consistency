# Agent Consistency

Methods for measuring behavioural consistency and uncertainty in LLM agents. Two approaches are implemented:

- **RSD (Recursive Semantic Divergence)** — learns a metric over agent trajectories using a MICO-style network trained with n-step TD learning. Predicts agent behavioral consistency from trace dynamics.

---

## Setup

**Install dependencies:**

```bash
pip install torch numpy tqdm backoff openai cohere python-dotenv \
            scikit-learn scipy pandas matplotlib seaborn
```

**Configure environment variables** in a `.env` file at the repo root:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
```

Both modules use Azure OpenAI for embeddings (via the Cohere `embed-v-4-0` model routed through Azure) and for LLM calls.

---

## Data

Traces are expected at `traces/<dataset>/<model>/` as JSON files named:

```
trajectories_<run_id>_<sample_id>.json
```

Each file contains a list of conversations, where each conversation is a list of turns with `role` (`user`, `agent`, `tool`) and `content` fields.

The test script also requires a `reward_dict.pkl` in the traces directory mapping trace names to progress rates (0.0–1.0).

Supported datasets: `tau2bench`, `toolsandbox`.

---

## RSD

All scripts run from the `rsd/` directory.

**1. Train the MICO network**

Reads traces from `traces/<dataset>/<model>/`, computes Cohere embeddings (cached to `embeddings/`), and trains for 1000 epochs. Uses an 80/20 train/test split by sample ID.

```bash
cd rsd
python train.py
```

Model checkpoint saved to `models/<dataset>/<model>/checkpoints_.../rsd_final.pth`.

**2. Evaluate on the held-out test set**

Loads the trained model and scores each test trajectory. Requires `reward_dict.pkl` in the traces directory.

```bash
python test.py
```

Results saved to `results/<dataset>/<model>/.../distance_scores.json`.