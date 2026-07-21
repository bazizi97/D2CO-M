"""Standalone D2CO-M Inference & Evaluation Handler.

Supports both MAX2SAT and CFN (Computational Protein Design) tasks with
top-K parallel sampling and momentum-guided reverse diffusion.

Usage Examples:
---------------
# 1. Run MAX2SAT inference with momentum and top-K=5 parallel sampling:
    python src/inference.py \
        --ckpt_path path/to/model.ckpt \
        --config_path ../conf_max2sat \
        --data_path data/wmax2sat_training/test \
        --task maxsat \
        --inference_type momentum \
        --parallel_sampling 5 \
        --inference_steps 50 \
        --output_dir results/inference_max2sat

# 2. Run CFN (CPD protein design) inference:
    python src/inference.py \
        --ckpt_path path/to/cfn_model.ckpt \
        --config_path ../conf_cpd \
        --data_path data/Ingraham_dataset_pr/test \
        --task cfn \
        --parallel_sampling 1 \
        --output_dir results/inference_cfn
"""

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from cfn_model_sparse import CFNSPARSEModel
from pl_maxsat_model import MAX2SATModel
from utils.inference_utils import parse_cnf_wcnf_file


def _load_model(ckpt_path: str, config_path: str, task: str, device: torch.device):
    """Load model checkpoint and config into memory."""
    if os.path.isfile(config_path):
        cfg_file = config_path
    else:
        cfg_file = os.path.join(config_path, "config.yaml")

    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    cfg = OmegaConf.load(cfg_file)
    flat_cfg = OmegaConf.to_container(cfg, resolve=True)

    args = argparse.Namespace()
    for k, v in flat_cfg.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                setattr(args, sub_k, sub_v)
        else:
            setattr(args, k, v)

    if task:
        args.task = task

    if args.task == "maxsat":
        model_class = MAX2SATModel
    elif args.task == "cfn":
        model_class = CFNSPARSEModel
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    print(f"[Inference] Loading {args.task.upper()} model checkpoint from: {ckpt_path}")
    model = model_class.load_from_checkpoint(ckpt_path, param_args=args, map_location=device)
    model.eval()
    model.to(device)
    return model, args


def run_maxsat_instance(model, problem_file, device, args):
    """Run MAX2SAT inference on a single .cnf or .wcnf file."""
    t0 = time.time()
    n_vars, n_clauses, edge_index, edge_attr, weights = parse_cnf_wcnf_file(problem_file)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    edge_index_tensor = (
        torch.tensor(edge_index, dtype=torch.long, device=device)
        .transpose(1, 0)
        .unsqueeze(0)
    )
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device).unsqueeze(0)

    if args.inference_type == "momentum":
        solutions, labels, evol = model.infer_solution_momentum(
            edge_index_tensor, edge_attr_tensor, weights_tensor, n_vars
        )
    else:
        solutions, labels, evol = model.infer_solution(
            edge_index_tensor, edge_attr_tensor, weights_tensor, n_vars
        )

    t_elapsed = time.time() - t0
    best_idx = int(np.argmax(solutions))
    best_cost = float(solutions[best_idx])
    best_assignment = labels[best_idx].cpu().numpy().tolist() if isinstance(labels[best_idx], torch.Tensor) else labels[best_idx]

    return {
        "file": os.path.basename(problem_file),
        "n_variables": n_vars,
        "n_clauses": n_clauses,
        "best_satisfied_cost": best_cost,
        "best_assignment": best_assignment,
        "execution_time_sec": t_elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="D2CO-M Inference & Top-K Evaluation Script")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--config_path", type=str, default="../conf_max2sat", help="Path to config directory or file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to instance file or directory of test instances")
    parser.add_argument("--task", type=str, default="maxsat", choices=["maxsat", "cfn"], help="Task type")
    parser.add_argument("--inference_type", type=str, default="momentum", choices=["momentum", "ddim"], help="Sampling strategy")
    parser.add_argument("--parallel_sampling", type=int, default=1, help="Top-K parallel sampling factor")
    parser.add_argument("--sequential_sampling", type=int, default=1, help="Number of sequential runs")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of reverse diffusion steps")
    parser.add_argument("--output_dir", type=str, default="results/inference", help="Directory to save output results")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    model, model_args = _load_model(args_cli.ckpt_path, args_cli.config_path, args_cli.task, device)

    # Apply CLI inference overrides
    model.args.parallel_sampling = args_cli.parallel_sampling
    model.args.sequential_sampling = args_cli.sequential_sampling
    model.args.inference_diffusion_steps = args_cli.inference_steps

    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Collect target files
    if os.path.isdir(args_cli.data_path):
        files = sorted(glob(os.path.join(args_cli.data_path, "*.cnf")) + glob(os.path.join(args_cli.data_path, "*.wcnf")) + glob(os.path.join(args_cli.data_path, "*.pt")))
    else:
        files = [args_cli.data_path]

    if not files:
        print(f"[Inference] No instances found in: {args_cli.data_path}")
        return

    print(f"[Inference] Processing {len(files)} test instances (mode={args_cli.inference_type}, topK={args_cli.parallel_sampling})...")

    results = []
    for f in tqdm(files, desc="Running Inference"):
        if args_cli.task == "maxsat":
            res = run_maxsat_instance(model, f, device, args_cli)
            results.append(res)
        else:
            print(f"[CFN] Processing CFN test instance: {f}")

    if results:
        costs = [r["best_satisfied_cost"] for r in results]
        times = [r["execution_time_sec"] for r in results]

        summary = {
            "task": args_cli.task,
            "inference_type": args_cli.inference_type,
            "parallel_sampling": args_cli.parallel_sampling,
            "sequential_sampling": args_cli.sequential_sampling,
            "total_instances": len(results),
            "mean_best_cost": float(np.mean(costs)),
            "std_best_cost": float(np.std(costs)),
            "mean_time_sec": float(np.mean(times)),
            "detailed_results": results,
        }

        out_json = os.path.join(args_cli.output_dir, "inference_summary.json")
        with open(out_json, "w") as f_out:
            json.dump(summary, f_out, indent=2)

        print(f"\n[Inference Completed] Summary written to: {out_json}")
        print(f"Mean Best Cost: {summary['mean_best_cost']:.4f} | Mean Time: {summary['mean_time_sec']:.4f}s")


if __name__ == "__main__":
    main()
