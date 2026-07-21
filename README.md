# D2CO-M: Discrete Graph Denoising Diffusion Model with Momentum-Based Sampling for Combinatorial Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch--lightning-2.0+-792ee5.svg)](https://www.pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

Neural network-based Combinatorial Optimization (CO) solvers are showing increasingly convincing performances on NP-complete (NPC) problems, alleviating the need for hand-crafted heuristics. These methods are usually tested on problems with very informative features, such as coordinates in the Euclidean Traveling Salesman Problem. In this paper, we introduce **D2CO-M**, a discrete graph denoising diffusion model that learns how to efficiently generate high-quality solutions using an original momentum-based sampling inference strategy. The dynamics of traditional score-matching denoising diffusion can be interpreted as a stochastic gradient ascent guided by the Stein score, the gradient of the log-likelihood. **D2CO-M** shifts from this simple gradient ascent to a momentum-based sampling strategy fitted for discrete diffusion, inspired by fast momentum-enhanced first-order methods, as exploited in the Adam optimizer. We experiment with this momentum-based variant on two NP-hard optimization problems with complex fine-grained features: random **Weighted Max-2-SAT** instances, and real-world **Cost Function Network** instances describing **Computational Protein Sequence Design** problems, establishing **D2CO-M** as a competitive and robust tool for complex, real-world combinatorial optimization.

---

## Key Features & Highlights

- **Discrete Graph Diffusion**: Categorical & binary graph denoising diffusion framework designed for complex, non-Euclidean combinatorial optimization instances.
- **Momentum-Based Inference Strategy**: Adam-inspired momentum tracking over discrete logit transitions to accelerate convergence and escape local minima during iterative reverse-diffusion sampling.
- **Problem Domains**:
  - **MAX-2-SAT / Weighted MAX-2-SAT**: Hard binary decision and optimization problems.
  - **Cost Function Networks (CFN)**: Computational Protein Sequence Design based on CATH protein backbones (20 amino acid categorical choices).
- **Decoupled Architecture**: Anisotropic GNN Encoder architecture.
- **Unified Training Pipeline**: Single Hydra-managed PyTorch Lightning training runner (`src/train.py`).
- **Parallel Benchmarking & Profiling**: Automated inference evaluation framework (`src/utils/inference_utils.py`) with GPU memory tracking and top-$k$ parallel sampling comparisons.

---

## Repository Structure

```text
D2CO-M/
├── README.md                          # Project documentation
├── pyproject.toml                     # Dependencies & environment configuration
├── data_scripts/
│   ├── load_Ingraham.sh               # Download CATH protein dataset for CFN
│   └── process_dataset.py             # Unified CNF generation, SPB solving, and graph JSON exporter
├── src/
│   ├── train.py                       # Unified PyTorch Lightning training script
│   ├── cfn_meta_sparse.py             # Base Lightning module for CFN graph models
│   ├── cfn_model_sparse.py             # Sparse CFN Lightning model (CATH / 20-class design)
│   ├── pl_meta_model.py               # Base Lightning module for MAX2SAT models
│   ├── pl_maxsat_model.py             # Lightning model for MAX2SAT binary diffusion
│   ├── co_datasets/
│   │   ├── Ingraham_dataset.py        # CATH & CFN protein graph dataset handlers
│   │   └── maxsat_dataset.py          # MAX2SAT sparse graph dataset handlers
│   ├── models/
│   │   ├── gnn_encoder.py             # Primary GNN Encoder architecture
│   │   ├── gnn_encoder_2.py           # Decoupled output dimension GNN Encoder
│   │   └── nn.py                      # SiLU, GroupNorm32, and timestep embedding layers
│   └── utils/
│       ├── diffusion_schedulers.py    # Categorical Diffusion & Inference step schedulers
│       ├── lr_schedulers.py           # Cosine & linear learning rate schedulers
│       ├── maxsat_utils.py            # MAX2SAT clause evaluation and graph edge converters
│       ├── effie_utils.py             # CFN energy scoring utilities
│       └── inference_utils.py         # DDIM & Momentum-based parallel inference profiling
└── conf/
    └── config.yaml                    # Hydra training & model configuration
```

---

## Installation & Setup

### Prerequisites
- Linux OS with NVIDIA GPU (CUDA 11.8+ / 12.0+)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Automatic Environment Setup

The repository provides an automated installation script (`install_env.sh`) that sets up the `.venv` environment, installs all dependencies declared in `pyproject.toml` (including `cnfgen`, `pybind11`, `setuptools`, and `torch`), activates the environment, and compiles `pyakmaxsat`:

```bash
# Clone repository
git clone https://github.com/bazizi97/D2CO-M.git
cd D2CO-M

# Make setup script executable and run setup
chmod +x install_env.sh
./install_env.sh

# Activate the virtual environment
source .venv/bin/activate
```

### Manual Environment Setup (Alternative)

If you prefer to run setup commands manually:

```bash
# 1. Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Sync dependencies from pyproject.toml
uv sync

# 3. Compile and install pyakmaxsat
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" CXXFLAGS="-include cstdint" pip install --no-build-isolation git+https://github.com/mullzhang/pyakmaxsat.git
```

---

## Dataset Download & Preparation

### 1. CATH Protein Dataset (Cost Function Network / CFN)
Download the Ingraham CATH dataset into your storage folder using the provided helper script:

```bash
# Download CATH dataset to ./data/cath
./data_scripts/load_Ingraham.sh ./data/cath
```

### 2. MAX-2-SAT / Weighted MAX-2-SAT Problems

Generate weighted MAX-2-SAT training (90k), validation (5k), and test (5k) sets solved with `akmaxsat`:

```bash
# Generate complete Weighted MAX-2-SAT dataset split
./data_scripts/generate_max2sat_dataset.sh ./data/max2sat 16
```

Or run custom single-directory generation with `data_scripts/process_dataset.py`:

```bash
# Custom generation: CNF -> WCNF -> Solve with Akmaxsat -> Export JSON graphs
python data_scripts/process_dataset.py --mode all --problem_dir ./data/maxsat --solve --solver akmaxsat --weighted --max_weight 10
```

---

## Training D2CO-M Models

Training is managed via `src/train.py` using **Hydra**. Problem-specific configurations are located in `conf_cpd` (Cost Function Network / Protein Design) and `conf_max2sat` (MAX-2-SAT).

### 1. Train Computational Protein Sequence Design (CPD / CFN) Model
Uses `conf_cpd` configuration:
```bash
python src/train.py --config-path ../conf_cpd task=cfn
```

### 2. Train MAX-2-SAT Model
Uses `conf_max2sat` configuration:
```bash
python src/train.py --config-path ../conf_max2sat task=maxsat
```

### Custom Hyper-parameter Overrides
Override parameters on the fly directly via CLI:
```bash
# Override learning rate, batch size, and diffusion steps for MAX-2-SAT
python src/train.py --config-path ../conf_max2sat task=maxsat model.learning_rate=1e-4 data.batch_size=32 model.diffusion_steps=1000
```

---

## Inference & Momentum-Based Sampling

D2CO-M supports evaluation via the standalone inference script `src/inference.py` or directly via Python API.

### 1. Command-Line Inference & Evaluation (`src/inference.py`)

Run top-$K$ parallel momentum-guided inference across test problem instances:

```bash
# Run MAX-2-SAT inference with momentum and top-K=5 parallel sampling
python src/inference.py \
    --ckpt_path path/to/model.ckpt \
    --config_path conf_max2sat \
    --data_path data/wmax2sat_training/test \
    --task maxsat \
    --inference_type momentum \
    --parallel_sampling 5 \
    --inference_steps 50 \
    --output_dir results/inference_max2sat
```

### 2. Python API

Load model checkpoints programmatically and run momentum-enhanced sampling:

```python
import torch
from pl_maxsat_model import MAX2SATModel

# Load trained checkpoint
model = MAX2SATModel.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Run momentum-enhanced inference on a MAX2SAT instance graph
best_cost, predictions, evol_sat = model.infer_solution_momentum(
    edge_index=edge_index,
    edge_attr=edge_attr,
    weights=weights,
    n_nodes=n_variables,
)

print(f"Optimal Satisfied Clause Weight: {best_cost}")
```

---

## License

This project is licensed under the [MIT License](LICENSE).

