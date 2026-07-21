"""Unified dataset generation, solving, and graph parsing pipeline for MAX2SAT problems.

Replaces data_generator.py, solve_instance.py, generates_problems.py, and generates_problems_spb.py.

Usage examples:
---------------
# 1. Generate random k-SAT CNF instances:
    python process_dataset.py --mode generate_cnf --n_problem 100 --out_dir ./data/cnf_instances

# 2. Process and solve existing .cnf files with SPB solver:
    python process_dataset.py --mode process --problem_dir ./data/cnf_instances --solve --solver spb --weighted

# 3. Complete pipeline (Generate + Solve + Convert to JSON graph representation):
    python process_dataset.py --mode all --problem_dir ./data/dataset --solve --solver spb --weighted
"""

import os
import sys
import time
import json
import random
import logging
import subprocess
from glob import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

try:
    import cnfgen
    HAS_CNFGEN = True
except ImportError:
    HAS_CNFGEN = False


# ==============================================================================
# 1. CNF Instance Generation (Adapted from data_generator.py)
# ==============================================================================

class KSATGenerator:
    """Generates random k-SAT CNF instances using cnfgen."""

    def __init__(self, min_n: int = 100, max_n: int = 100, min_k: int = 2, max_k: int = 2,
                 min_alpha: float = 8.0, max_alpha: float = 8.0):
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def create_random_instance(self, cnf_seed: int) -> list:
        if not HAS_CNFGEN:
            raise ImportError("cnfgen library is required for CNF generation. Install via `pip install cnfgen`.")
        
        k = np.random.randint(self.min_k, self.max_k + 1)
        n = np.random.randint(self.min_n, self.max_n + 1)
        alpha = np.random.uniform(self.min_alpha, self.max_alpha)
        m = max(int(np.ceil(n * alpha)), 1)
        cnf = cnfgen.RandomKCNF(k, n, m, seed=cnf_seed)

        clauses = [np.int64(cls) for cls in cnf.clauses()]
        return clauses


def write_dimacs_cnf(f: list, path: str):
    """Save a CNF formula into standard DIMACS CNF format."""
    num_v = int(np.max([np.max(np.abs(clause)) for clause in f]))
    num_c = len(f)
    with open(path, 'w') as file:
        file.write(f'p cnf {num_v} {num_c}\n')
        for clause in f:
            line = ' '.join(str(l) for l in clause) + ' 0\n'
            file.write(line)


def generate_cnf_instances(out_dir: str, args):
    """Generate multiple random CNF problem instances."""
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(args.seed)
    seeds = np.random.randint(-2**31, 2**31 - 1, args.n_problem)

    generator = KSATGenerator(
        min_n=args.min_n, max_n=args.max_n,
        min_k=args.min_k, max_k=args.max_k,
        min_alpha=args.min_alpha, max_alpha=args.max_alpha
    )

    print(f"[Generator] Generating {args.n_problem} CNF instances in: {out_dir}")
    for idx, seed in tqdm(enumerate(seeds), total=len(seeds), desc="Generating CNF"):
        instance = generator.create_random_instance(cnf_seed=int(seed))
        out_path = os.path.join(out_dir, f"{idx}.cnf")
        write_dimacs_cnf(instance, out_path)


# ==============================================================================
# 2. Format Conversion & Feature Extraction
# ==============================================================================

def cnf2wcnf(problem: str, max_weight: int = 10, weighted: bool = True):
    """Convert a .cnf DIMACS file into a .wcnf file."""
    if not problem.endswith(".cnf"):
        raise ValueError("Input file must have a .cnf extension.")

    new_file_path = problem.rsplit(".cnf", 1)[0] + ".wcnf"
    n_var = 0

    with open(problem, "r") as file, open(new_file_path, "w") as new_file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith("p"):
                parts = line.split()
                n_var = int(parts[2])
                if len(parts) >= 4 and parts[1] == "cnf":
                    parts[1] = "wcnf"
                new_line = " ".join(parts) + " 10000000000000.000\n"
                new_file.write(new_line)

            elif line.startswith("c"):
                new_file.write(line + "\n")

            else:
                weight = random.randint(1, max_weight) if weighted else 1
                new_file.write(f"{weight} {line}\n")

    return new_file_path, n_var


def process_wcnf_file(wcnf_file: str):
    """Parse WCNF file into edge indices, 4D edge attributes, clauses F, and weights."""
    edge_attr = []
    edge_index = []
    weights = []
    F = []

    with open(wcnf_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("c") or line.startswith("p"):
                continue
            parts = line.split()
            if not parts:
                continue

            weight = int(float(parts[0]))
            lit1 = int(parts[1])
            lit2 = int(parts[2])

            edge_index.append([abs(lit1) - 1, abs(lit2) - 1])
            F.append([lit1, lit2])
            weights.append(weight)

            # Map literal sign combinations to 4D one-hot edge attributes
            if lit1 < 0 and lit2 > 0:
                edge_attr.append([1, 0, 1, 1])
            elif lit1 > 0 and lit2 > 0:
                edge_attr.append([0, 1, 1, 1])
            elif lit1 > 0 and lit2 < 0:
                edge_attr.append([1, 1, 0, 1])
            elif lit1 < 0 and lit2 < 0:
                edge_attr.append([1, 1, 1, 0])
            else:
                raise ValueError(f"Invalid clause literals: {lit1}, {lit2}")

    return edge_index, edge_attr, F, weights


def count_satisfied_clauses(F: list, assignment: list, weights: list = None) -> int:
    """Compute total satisfied clause weights for a variable assignment."""
    if weights is None:
        weights = [1] * len(F)
    count = 0
    for clause, w in zip(F, weights):
        var1, var2 = clause
        val1 = assignment[abs(var1) - 1]
        val2 = assignment[abs(var2) - 1]
        if (var1 > 0 and val1) or (var1 < 0 and not val1) or \
           (var2 > 0 and val2) or (var2 < 0 and not val2):
            count += w
    return count


# ==============================================================================
# 3. Solver Engines (SPB-MaxSAT & AKMaxSAT)
# ==============================================================================

def run_spb_solver(wcnf_path: str, n_var: int, solver_path: str, timeout: float):
    """Run SPB-MaxSAT solver binary on a WCNF problem file with timeout."""
    if not os.path.isfile(wcnf_path):
        print(f"   [SPB] WARNING: .wcnf file not found: {wcnf_path}")
        return None, float("nan")

    cmd = ["timeout", "-s", "15", str(timeout), solver_path, wcnf_path]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        print(f"   [SPB] ERROR running solver: {exc}")
        return None, float("nan")
    wall_time = time.perf_counter() - t0

    solutions = []
    for line in result.stdout.splitlines():
        if "vsol" in line:
            parts = line.split()
            try:
                bits = [int(b) for b in list(parts[-1])]
                solutions.append((float(parts[0]), bits))
            except (ValueError, IndexError):
                continue

    if not solutions:
        return None, wall_time

    _, best_bits = solutions[-1]
    if len(best_bits) != n_var:
        best_bits = (best_bits + [0] * n_var)[:n_var]

    return best_bits, wall_time


def run_akmaxsat_solver(cnf_path: str, n_var: int):
    """Run pyakmaxsat solver on a CNF problem file."""
    try:
        from pyakmaxsat import AKMaxSATSolver
        solver = AKMaxSATSolver()
        sol = solver.sample_wcnf(cnf_path)
        nodes = [1 if x > 0 else 0 for x in sol]
        return (nodes + [0] * n_var)[:n_var], 0.05
    except Exception as exc:
        print(f"   [AKMaxSAT] Warning/Error running solver: {exc}")
        return None, float("nan")


# ==============================================================================
# 4. Pipeline Execution Helper
# ==============================================================================

def process_single_instance(problem_file: str, args):
    """Process a single .cnf file: converts to .wcnf, solves, and saves .json graph."""
    max_w = args.max_weight if args.weighted else 1
    wcnf_file, n_var = cnf2wcnf(problem_file, max_weight=max_w, weighted=args.weighted)

    t1 = time.time()
    if args.solve:
        if args.solver == "spb":
            nodes, solver_time = run_spb_solver(
                wcnf_path=wcnf_file,
                n_var=n_var,
                solver_path=args.solver_path,
                timeout=args.timeout,
            )
        elif args.solver == "akmaxsat":
            nodes, solver_time = run_akmaxsat_solver(cnf_path=problem_file, n_var=n_var)
        else:  # random
            nodes = [random.choice([0, 1]) for _ in range(n_var)]
            solver_time = 0.0

        if nodes is None:
            # Fallback to random if solver returns no solution
            nodes = [random.choice([0, 1]) for _ in range(n_var)]
    else:
        nodes = [random.choice([0, 1]) for _ in range(n_var)]
        solver_time = 0.0
    t2 = time.time()

    edge_index, edge_attr, F, weights = process_wcnf_file(wcnf_file)

    assignment_bool = [True if ele == 1 else False for ele in nodes]
    sat_count = count_satisfied_clauses(
        F, assignment_bool, weights=weights if args.weighted else None
    )

    data = {
        "nodes": nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "weights": weights if args.weighted else [1] * len(edge_index),
        "satisfied_clauses": sat_count,
        "execution_time": solver_time if args.solve else (t2 - t1)
    }

    json_output_path = wcnf_file + ".json"
    with open(json_output_path, "w") as f:
        json.dump(data, f)


def process_dataset(problem_dir: str, args):
    """Process all .cnf files in problem_dir in parallel."""
    problems_files = glob(os.path.join(problem_dir, "*.cnf"))
    if not problems_files:
        print(f"No .cnf files found in {problem_dir}")
        return

    print(f"[DatasetProcessor] Processing {len(problems_files)} .cnf files in parallel (workers={args.n_workers})...")
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(process_single_instance, f, args): f for f in problems_files}
        for future in tqdm(as_completed(futures), total=len(problems_files), desc="Processing instances"):
            try:
                future.result()
            except Exception as e:
                file_path = futures[future]
                print(f"Error processing {file_path}: {e}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="Unified MAX2SAT Data Generation & Processing Tool")
    parser.add_argument("--mode", type=str, default="process", choices=["generate_cnf", "process", "all"],
                        help="Mode: generate_cnf, process, or all")
    
    # CNF Generation options
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--min_n", type=int, default=100, help="Minimum number of variables")
    parser.add_argument("--max_n", type=int, default=100, help="Maximum number of variables")
    parser.add_argument("--min_k", type=int, default=2, help="Minimum arity (k-SAT)")
    parser.add_argument("--max_k", type=int, default=2, help="Maximum arity (k-SAT)")
    parser.add_argument("--min_alpha", type=float, default=8.0, help="Minimum clause/variable ratio")
    parser.add_argument("--max_alpha", type=float, default=8.0, help="Maximum clause/variable ratio")
    parser.add_argument("--n_problem", type=int, default=200, help="Number of problems to generate")

    # Dataset Processing & Solving options
    parser.add_argument("--problem_dir", type=str, default="./data", help="Directory containing .cnf files")
    parser.add_argument("--solve", action="store_true", help="Solve instances using designated solver")
    parser.add_argument("--solver", type=str, default="spb", choices=["spb", "akmaxsat", "random"],
                        help="Solver engine to use")
    parser.add_argument("--solver_path", type=str,
                        default="./solvers/SPB-MaxSAT",
                        help="Path to SPB solver binary")
    parser.add_argument("--weighted", action="store_true", help="Generate weighted WCNF instances")
    parser.add_argument("--max_weight", type=int, default=10, help="Maximum soft clause weight")
    parser.add_argument("--timeout", type=float, default=5.0, help="Solver timeout (seconds)")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of thread pool workers")

    args = parser.parse_args()

    if args.mode in ["generate_cnf", "all"]:
        generate_cnf_instances(args.problem_dir, args)

    if args.mode in ["process", "all"]:
        process_dataset(args.problem_dir, args)
