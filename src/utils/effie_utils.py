"""Cost Function Network (CFN) Energy & Native Sequence Recovery (NSR) Utilities.

Provides functions to compute Native Sequence Recovery (NSR), masked NSR,
pairwise energy matrix scores (Effie scores), and KL-divergence logit losses
for protein design and CFN optimization models.
"""

import torch
import torch.nn.functional as F


def remove_N_C_ter(crd: torch.Tensor, seq: str, chain_idx=None, tm=None, return_cut: bool = False):
    """Remove missing / unobserved residues at N-terminal and C-terminal ends.

    Args:
        crd: Coordinate tensor of shape (L, 4, 3) for backbone atoms.
        seq: Amino acid sequence string of length L.
        chain_idx: Optional chain identifier array.
        tm: Optional backbone mask array.
        return_cut: If True, returns a tuple containing the trimmed sequence parts.

    Returns:
        Tuple of trimmed coordinates, sequence, chain indices, and masks.
    """
    cut = [[], []]  # Stores cut amino acids [N-ter, C-ter]

    # Trim N-terminal missing residues (zero or NaN coordinates)
    while torch.sum(torch.abs(crd[0, :4])) == 0 or torch.isnan(torch.sum(torch.abs(crd[0, :4]))):
        crd = crd[1:]
        cut[0].append(seq[0])
        seq = seq[1:]
        if chain_idx is not None:
            chain_idx = chain_idx[1:]
        if tm is not None:
            tm = tm[1:]

    # Trim C-terminal missing residues
    while torch.sum(torch.abs(crd[-1, :4])) == 0 or torch.isnan(torch.sum(torch.abs(crd[-1, :4]))):
        crd = crd[:-1]
        cut[1].append(seq[-1])
        seq = seq[:-1]
        if chain_idx is not None:
            chain_idx = chain_idx[:-1]
        if tm is not None:
            tm = tm[:-1]

    cut[1] = [cut[1][i] for i in range(-1, -len(cut[1]) - 1, -1)]
    cut = [''.join(cut[0]), ''.join(cut[1])]

    if return_cut:
        return (crd, seq, chain_idx, tm, cut)
    else:
        return (crd, seq, chain_idx, tm)


def compute_nsr(seq_nat: torch.Tensor, seq_pred: torch.Tensor) -> torch.Tensor:
    """Compute Native Sequence Recovery (NSR) percentage across sequences.

    Args:
        seq_nat: Ground-truth amino acid sequence tensor.
        seq_pred: Predicted amino acid sequence tensor.

    Returns:
        NSR percentage (float scalar tensor).
    """
    matched_aa = (seq_nat == seq_pred)
    return torch.sum(matched_aa) * 100.0 / seq_nat.shape[-1]


def compute_masked_nsr(seq_nat: torch.Tensor, seq_pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute Native Sequence Recovery (NSR), ignoring masked residues (where mask == True).

    Args:
        seq_nat: Ground-truth class indices of shape (L,) or (B, L).
        seq_pred: Predicted class indices of shape (L,) or (B, L).
        mask: Boolean mask of same shape (True = masked out, False = valid residue).

    Returns:
        NSR percentage tensor computed over valid positions.
    """
    assert seq_nat.shape == seq_pred.shape == mask.shape, "All input shapes must match"

    valid = ~mask  # False in mask -> valid position
    matched = (seq_nat == seq_pred) & valid
    n_valid = valid.sum(dim=-1).clamp(min=1)  # Avoid division by zero
    nsr = matched.sum(dim=-1).float() * 100.0 / n_valid

    return nsr


def compute_kl_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence loss between predicted logits and target one-hot class indices.

    Args:
        logits: Logit predictions tensor of shape (B, K).
        target: Target class integer indices of shape (B,).

    Returns:
        Scalar KL divergence loss averaged across batch.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    target_probs = F.one_hot(target, num_classes=logits.size(-1)).float()

    kl = F.kl_div(log_probs, target_probs, reduction='batchmean', log_target=False)
    return kl


def compute_effie_score(seq: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch_size: int = 1) -> float:
    """Compute Effie energy score of an amino acid sequence given graph interaction matrices.

    Args:
        seq: Predicted discrete sequence tensor of shape (V,).
        edge_index: Graph edge indices of shape (2, E).
        edge_attr: Pairwise 20x20 energy attribute matrices of shape (E, 400).
        batch_size: Batch scaling factor.

    Returns:
        Total energy score of the sequence configuration.
    """
    i = edge_index[0]
    j = edge_index[1]

    # Filter unique pairwise edges (i < j)
    mask = i < j
    i = i[mask]
    j = j[mask]
    edge_subset = edge_attr[mask]

    # Extract residue choices at source and target nodes
    a_i = seq[i]
    a_j = seq[j]

    # Reshape edge_attr to (E_sub, 20, 20) matrix and index pairwise energy entries
    scores = edge_subset.view(-1, 20, 20)[torch.arange(edge_subset.size(0)), a_i, a_j]

    return scores.sum().item() / batch_size