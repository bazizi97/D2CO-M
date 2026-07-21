"""MAX-2-SAT Graph Transformation & Clause Evaluation Utilities.

Provides helper routines for parsing CNF/WCNF clause formulas into graph edge
representations, converting edge indices/attributes to dense adjacency matrices,
and computing total satisfied clause weights for binary assignment solutions.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


def process_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor):
    """Construct bidirectional edge indices and swap literal flags for reversed edges.

    Args:
        edge_index: Tensor of shape (B, 2, E) containing directed edge pairs.
        edge_attr: Tensor of shape (B, E, 4) containing clause literal attributes.

    Returns:
        edges_with_reversed: Concatenated original and reversed edge indices of shape (B, 2, 2E).
        edge_attr_full: Concatenated original and swapped edge attributes of shape (B, 2E, 4).
    """
    # Create reversed edges (swap node i and node j)
    reversed_edges = edge_index.flip(1)

    # Concatenate original and reversed edges
    edges_with_reversed = torch.cat([edge_index, reversed_edges], dim=2)

    reversed_edge_attr = edge_attr.clone()

    # Identify edges where literal signs differ (col 1 != col 2)
    swap_mask = edge_attr[:, :, 1] != edge_attr[:, :, 2]

    col1 = edge_attr[:, :, 1]
    col2 = edge_attr[:, :, 2]

    # Swap literal sign flags for reversed edges
    reversed_edge_attr[:, :, 1] = torch.where(swap_mask, col2, col1)
    reversed_edge_attr[:, :, 2] = torch.where(swap_mask, col1, col2)

    edge_attr_full = torch.cat([edge_attr, reversed_edge_attr], dim=1)

    return edges_with_reversed, edge_attr_full


def process_edges_sparse(edge_index: torch.Tensor, edge_attr: torch.Tensor, weights: torch.Tensor, batch_vector: torch.Tensor = None):
    """Build bidirectional edge set and scale edge_attr by normalized clause weights.

    When ``batch_vector`` is provided (node -> graph index, shape ``(V,)``), each edge weight
    is normalized by the maximum weight of its own graph.

    Args:
        edge_index: Tensor of shape [2, n_edges]
        edge_attr: Tensor of shape [n_edges, 4]
        weights: Tensor of shape [n_edges]
        batch_vector: Tensor of shape [V,] or None

    Returns:
        edges_with_reversed: Bidirectional edge indices of shape [2, 2 * n_edges]
        edge_attr_final: Scaled edge attributes of shape [2 * n_edges, 4]
    """
    if batch_vector is not None:
        edge_graph = batch_vector[edge_index[0]]
        n_graphs = int(edge_graph.max().item()) + 1

        w_max = torch.zeros(n_graphs, dtype=weights.dtype, device=weights.device)
        w_max.scatter_reduce_(0, edge_graph, weights, reduce="amax", include_self=True)
        w_max = w_max.clamp(min=1e-8)

        weights_normalized = weights / w_max[edge_graph]
    else:
        weights_normalized = normalize_by_max(weights)

    weights_normalized = weights_normalized.repeat(2)

    reversed_edges = edge_index.flip(0)
    edges_with_reversed = torch.cat([edge_index, reversed_edges], dim=1)

    reversed_edge_attr = edge_attr.clone()
    swap_mask = edge_attr[:, 1] != edge_attr[:, 2]
    col1 = edge_attr[:, 1]
    col2 = edge_attr[:, 2]
    reversed_edge_attr[:, 1] = torch.where(swap_mask, col2, col1)
    reversed_edge_attr[:, 2] = torch.where(swap_mask, col1, col2)

    edge_attr_full = torch.cat([edge_attr, reversed_edge_attr], dim=0)
    edge_attr_final = edge_attr_full * weights_normalized.unsqueeze(1)

    return edges_with_reversed, edge_attr_final


def process_clauses(clauses: list):
    """Convert raw MAX-2-SAT 2-literal clause lists into edge indices and 4D attributes.

    Args:
        clauses: List of 2-element signed integer lists [lit1, lit2].

    Returns:
        edge_index: Tensor of shape (2, E).
        edge_attr: Tensor of shape (E, 4).
    """
    edge_index = []
    edge_attr = []
    for clause in clauses:
        edge_index.append([abs(int(clause[0])) - 1, abs(int(clause[1])) - 1])
        if clause[0] > 0 and clause[1] > 0:
            edge_attr.append([0, 0, 0, 1])
        elif clause[0] < 0 and clause[1] > 0:
            edge_attr.append([0, 1, 0, 0])
        elif clause[0] > 0 and clause[1] < 0:
            edge_attr.append([0, 0, 1, 0])
        elif clause[0] < 0 and clause[1] < 0:
            edge_attr.append([1, 0, 0, 0])
    return torch.tensor(edge_index), torch.tensor(edge_attr)


def edge_to_adj(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int):
    """Convert sparse edge_index and edge_attr to a dense adjacency matrix.

    Args:
        edge_index: Tensor of shape [B, 2, E] containing edge indices.
        edge_attr: Tensor of shape [B, E, N] containing edge attributes.
        num_nodes: Total number of nodes V.

    Returns:
        Dense adjacency matrix of shape [B, V, V, N].
    """
    adj_matrices = []
    edge_index, edge_attr = process_edges(edge_index=edge_index, edge_attr=edge_attr)

    if edge_index.shape[0] == 1:
        return to_dense_adj(
            edge_index[0], edge_attr=edge_attr[0], max_num_nodes=num_nodes
        )
    else:
        for i in range(edge_index.shape[0]):
            adj = to_dense_adj(
                edge_index[i], edge_attr=edge_attr[i], max_num_nodes=num_nodes
            )
            adj_matrices.append(adj.squeeze(0))

        return torch.stack(adj_matrices)


def count_satisfied_clauses(assignment: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, edge_weights: torch.Tensor = None) -> float:
    """Compute mean sum of satisfied clause weights for a batch of binary assignments.

    Args:
        assignment: Binary variable assignments tensor of shape (B, V).
        edge_index: Edge index tensor of shape (B, 2, E).
        edge_attr: 4D clause attribute tensor of shape (B, E, 4).
        edge_weights: Optional clause weight tensor of shape (B, E) or (E,).

    Returns:
        Mean total satisfied clause weight per graph across the batch.
    """
    batch_size, n_nodes = assignment.shape
    _, _, n_edges = edge_index.shape

    i_nodes = edge_index[:, 0, :]
    j_nodes = edge_index[:, 1, :]

    i_values = torch.gather(assignment, 1, i_nodes)
    j_values = torch.gather(assignment, 1, j_nodes)

    # Encode binary pairs (j, i) into 4-class categorical index: 2*j + i
    clause_indices = (j_values * 2 + i_values).long()

    batch_idx = torch.arange(batch_size).view(-1, 1).expand(-1, n_edges)
    edge_idx = torch.arange(n_edges).view(1, -1).expand(batch_size, -1)

    satisfied = edge_attr[batch_idx, edge_idx, clause_indices]

    if edge_weights is not None:
        satisfied = satisfied * edge_weights

    n_satisfied_per_graph = satisfied.sum(dim=1).float().mean().item()
    return n_satisfied_per_graph


def reshape_with_point_indicator(flat_tensor: torch.Tensor, point_indicator: torch.Tensor) -> torch.Tensor:
    """Reshape a flat batched node tensor into a padded 2D tensor of shape [batch_size, max_nodes].

    Args:
        flat_tensor: Flat node feature tensor of shape [sum(nodes)].
        point_indicator: Node counts per batch graph of shape [batch_size].

    Returns:
        Padded 2D tensor of shape [batch_size, max_nodes].
    """
    if point_indicator.sum() != flat_tensor.size(0):
        raise ValueError("Sum of point_indicator does not match flat_tensor size.")

    max_nodes = point_indicator.max().item()
    batch_size = point_indicator.size(0)
    result_tensor = torch.zeros(batch_size, max_nodes, device=flat_tensor.device)

    start_idx = 0
    for i in range(batch_size):
        num_nodes = int(point_indicator[i].item())
        result_tensor[i, :num_nodes] = flat_tensor[start_idx: start_idx + num_nodes]
        start_idx += num_nodes

    return result_tensor


def normalize_by_max(w: torch.Tensor) -> torch.Tensor:
    """Normalize weight tensor by dividing by its maximum value."""
    return w / (w.max() + 1e-8)


def compute_run_csp_loss(logits: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    """Compute differentiable soft energy loss over Constraint Satisfaction Network graphs.

    Args:
        logits: Softmax logit predictions of shape (V, C).
        edge_index: Global graph edge index tensor of shape (2, E).
        edge_attr: Constraint matrix attributes per edge of shape (E, C * C).

    Returns:
        Differentiable scalar negative log-satisfaction energy loss.
    """
    V, C = logits.shape
    E = edge_index.shape[1]

    # Compute soft assignment probabilities phi (V, C)
    phi = F.softmax(logits, dim=-1)

    src = edge_index[0]
    dst = edge_index[1]

    phi_x = phi[src]  # (E, C)
    phi_y = phi[dst]  # (E, C)

    # Reshape edge attribute to (E, C, C) bilinear constraint tensor
    A = edge_attr.view(E, C, C)

    # Bilinear satisfaction calculation: p_sat = phi_x^T * A * phi_y
    Ay = torch.einsum('ecf,ef->ec', A, phi_y)
    p_sat = torch.einsum('ec,ec->e', phi_x, Ay)

    p_sat = p_sat.clamp(min=1e-8)
    loss = -torch.log(p_sat).mean()

    return loss