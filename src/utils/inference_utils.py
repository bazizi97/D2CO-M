import torch
import time
from pl_maxsat_model import MAX2SATModel
import argparse
import numpy as np
import json


def parse_cnf_wcnf_file(filepath):
    """
    Parse both CNF and WCNF files.
    
    Parameters:
    -----------
    filepath : str
        Path to the CNF or WCNF file
    
    Returns:
    --------
    tuple: (n_variables, n_clauses, clauses, weights)
        - n_variables: int, number of variables
        - n_clauses: int, number of clauses
        - clauses: list of lists, each inner list contains literals (integers)
        - weights: list of floats/ints, weight for each clause
                   (all 1.0 for CNF, actual weights for WCNF)
    
    Format specifications:
    ----------------------
    CNF:  p cnf <n_variables> <n_clauses>
          <literal> <literal> ... 0
    
    WCNF: p wcnf <n_variables> <n_clauses> [<top_weight>]
          <weight> <literal> <literal> ... 0
    """
    clauses = []
    weights = []
    n_variables = n_clauses = top_weight = None
    is_wcnf = False

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('c'):
                continue

            # Parse the problem line
            if line.startswith('p'):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Invalid problem line: {line}")
                
                format_type = parts[1].lower()
                
                if format_type == 'cnf':
                    is_wcnf = False
                    n_variables = int(parts[2])
                    n_clauses = int(parts[3])
                elif format_type == 'wcnf':
                    is_wcnf = True
                    n_variables = int(parts[2])
                    n_clauses = int(parts[3])
                    if len(parts) > 4:
                        top_weight = int(parts[4])
                else:
                    raise ValueError(f"Unknown format: {format_type}. Expected 'cnf' or 'wcnf'")
                continue

            # Parse clause line
            parts = list(map(int, line.split()))
            
            if is_wcnf:
                # WCNF format: <weight> <literal> ... 0
                if len(parts) < 2:
                    continue
                weight = parts[0]
                literals = parts[1:]
                if literals[-1] == 0:
                    literals = literals[:-1]
                weights.append(weight)
                clauses.append(literals)
            else:
                # CNF format: <literal> ... 0
                literals = parts
                if literals[-1] == 0:
                    literals = literals[:-1]
                weights.append(1)  # Default weight for CNF
                clauses.append(literals)

    # Validation
    if n_clauses is not None and len(clauses) != n_clauses:
        print(f"Warning: Declared {n_clauses} clauses, but found {len(clauses)} in file.")

    if len(clauses) != len(weights):
        raise ValueError(f"Mismatch: {len(clauses)} clauses but {len(weights)} weights")
    
    edge_index, edge_attr = process_clauses(clauses=clauses)

    return n_variables, n_clauses, edge_index, edge_attr, weights

def process_clauses(clauses):
    
    edge_index = []
    edge_attr = []
    for clause in clauses:
        edge_index.append([abs(int(clause[0])) - 1, abs(int(clause[1])) - 1])
        if int(clause[0]) < 0 and int(clause[1]) > 0:
            edge_attr.append([1, 0, 1, 1])
        elif int(clause[0]) > 0 and int(clause[1]) > 0:
            edge_attr.append([0, 1, 1, 1])
        elif int(clause[0]) > 0 and int(clause[1]) < 0:
            edge_attr.append([1, 1, 0, 1])
        elif int(clause[0]) < 0 and int(clause[1]) < 0:
            edge_attr.append([1, 1, 1, 0])
        else:
            raise Exception(f"Problem with clause {int(clause[0])}{int(clause[1])}")
    
    return edge_index, edge_attr

def run_inference(
    path_checkpoints,
    config_file,
    problem_file,
    n_samples=5,
    inference_type="ddim",
    inference_steps=50,
):

    with open(config_file, "r") as f:
        config_dict = json.load(f)

    config = argparse.Namespace(**config_dict)

    if torch.cuda.is_available():
        # Get the CUDA device
        device = torch.device("cuda")

    if config.task == "maxsat":
        model_class = MAX2SATModel
    else:
        raise NotImplementedError

    model = model_class.load_from_checkpoint(path_checkpoints, param_args=config)

    time.sleep(0.02)
    t1 = time.time()
    n_variables, n_clauses, edge_index, edge_attr, weights = parse_cnf_wcnf_file(problem_file)

    
    weights = torch.from_numpy(np.array(weights, dtype=np.float32))

    model.args.sequential_sampling = n_samples
    model.args.inference_diffusion_steps = inference_steps
    if inference_type == "ddim":
        sol = model.infer_solution(
            torch.tensor(edge_index)
            .transpose(1, 0)
            .unsqueeze(0)
            .to(device),
            torch.tensor(edge_attr).unsqueeze(0).to(device),
            weights.to(device),
            n_variables,
        )
    elif inference_type == "momentum":
        sol = model.infer_solution_momentum(
            torch.tensor(edge_index)
            .transpose(1, 0)
            .unsqueeze(0)
            .to(device),
            torch.tensor(edge_attr).unsqueeze(0).to(device),
            weights.to(device),
            n_variables,
        )
    else:
        NotImplementedError

    best_sol_idx = np.argmax(sol[0])

    best_sat = sol[0][best_sol_idx]
    best_sol = sol[1][best_sol_idx]

    t2 = time.time()

    return best_sat, best_sol, t2 - t1, n_variables, n_clauses
