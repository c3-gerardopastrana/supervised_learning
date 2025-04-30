import torch

def compute_wandb_metrics(outputs, sigma_w_inv_b):
    """
    Computes and returns a dictionary of metrics to be logged to wandb.

    Args:
        outputs (torch.Tensor): Eigenvalues or similar vector.
        sigma_w_inv_b (torch.Tensor): A matrix used for trace, rank, and condition number.

    Returns:
        dict: A dictionary of computed metrics.
    """
    eps = 1e-10

    eigvals_norm = outputs / outputs.sum()
    max_eigval_norm = eigvals_norm.max().item()
    min_eigval_norm = eigvals_norm.min().item()
    quantile_25 = torch.quantile(eigvals_norm, 0.25).item()
    quantile_50 = torch.quantile(eigvals_norm, 0.5).item()
    quantile_75 = torch.quantile(eigvals_norm, 0.75).item()

    eigvals_norm = torch.clamp(eigvals_norm, min=eps, max=1.0)
    eigvals_norm /= eigvals_norm.sum()
    entropy = -(eigvals_norm * eigvals_norm.log()).sum().item()



    trace = torch.trace(sigma_w_inv_b).item()
    rank_sigma = torch.linalg.matrix_rank(sigma_w_inv_b).item()
    condition_sigma = torch.linalg.cond(sigma_w_inv_b).item()
    off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
    sum_squared_off_diag = torch.sum(off_diag ** 2).item()
    diag_var = torch.var(torch.diagonal(sigma_w_inv_b)).item()


    metrics = {
        "entropy": entropy,
        "max_eigval_norm": max_eigval_norm,
        "min_eigval_norm": min_eigval_norm,
        "quantile_25": quantile_25,
        "quantile_50": quantile_50,
        "quantile_75": quantile_75,
        "trace_sigma": trace,
        "rank_sigma": rank_sigma,
        "condition_sigma": condition_sigma,
        "sum_squared_off_diag": sum_squared_off_diag,
        "diag_var": diag_var,
    }

    return metrics

