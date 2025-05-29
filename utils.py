import torch

def compute_wandb_metrics(Xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t, mu):
    """
    Computes and returns a dictionary of metrics to be logged to wandb.

    Args:
        outputs (torch.Tensor): Eigenvalues or similar vector.
        sigma_w_inv_b (torch.Tensor): A matrix used for trace, rank, and condition number.

    Returns:
        dict: A dictionary of computed metrics.
    """
    eps = 1e-10

    evals_complex, evecs_complex = torch.linalg.eig(sigma_w_inv_b)
    tol = 1e-6 # Tolerance for considering imaginary part zero
    is_complex = torch.abs(evals_complex.imag) > tol
    complex_count = torch.sum(is_complex).item() # Flag if *any* eigenvalue was complex beyond tolerance

    real_idx = ~is_complex
    evals = evals_complex[real_idx].real 

    eigvals_norm = evals / evals.sum()
    max_eigval_norm = eigvals_norm.max().item()
    min_eigval_norm = eigvals_norm.min().item()
    quantile_25 = torch.quantile(eigvals_norm, 0.25).item()
    quantile_50 = torch.quantile(eigvals_norm, 0.5).item()
    quantile_75 = torch.quantile(eigvals_norm, 0.75).item()

    eigvals_norm = torch.clamp(eigvals_norm, min=eps, max=1.0)
    eigvals_norm /= eigvals_norm.sum()
    entropy = -(eigvals_norm * eigvals_norm.log()).sum().item()

    entropy_w = spectral_entropy(sigma_w)
    entropy_b = spectral_entropy(sigma_b)
    entropy_t = spectral_entropy(sigma_t)
    




    trace = torch.trace(sigma_w_inv_b).item()
    rank_sigma = torch.linalg.matrix_rank(sigma_w_inv_b).item()
    condition_sigma = torch.linalg.cond(sigma_w_inv_b).item()
    off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
    sum_squared_off_diag = torch.sum(off_diag ** 2).item()
    diag_var = torch.var(torch.diagonal(sigma_w_inv_b)).item()
    diag_var_b = torch.var(torch.diagonal(sigma_w_inv_b)).item()
    diag_var_w = torch.var(torch.diagonal(sigma_w_inv_b)).item()
 
    trace_b = torch.trace(sigma_b).item()
    trace_w = torch.trace(sigma_w).item()
    
    off_diag_b = sigma_b - torch.diag(torch.diagonal(sigma_b))
    sum_squared_off_diag_b = torch.sum(off_diag_b ** 2).item()
    off_diag_w = sigma_w - torch.diag(torch.diagonal(sigma_w))
    sum_squared_off_diag_w = torch.sum(off_diag_w ** 2).item()


    # L2 norms of class means
    norms = torch.norm(Xc_mean, dim=1)
    
    # Mean and std of norms (do they all live on a sphere?)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    
    # Check if the class means themselves are centered around the origin
    #mean_of_means = Xc_mean.mean(dim=0)
    centered = torch.norm(mu).item()

    diff = (sigma_w + sigma_b) - sigma_t.to(torch.float32)
    max_diff = torch.max(torch.abs(diff)).item()



    metrics = {
        "entropy": entropy,
        "entropy_w": entropy_w,
        "entropy_b": entropy_b,
        "entropy_t": entropy_t,
        "complex_count":complex_count, 
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
        "diag_var_b": diag_var_b,
        "diag_var_w": diag_var_w,
        "trace_b": trace_b,
        "trace_w":trace_w,
        "sum_squared_off_diag_w":sum_squared_off_diag_w,
        "sum_squared_off_diag_b":sum_squared_off_diag_b,
        "mean_of_class_means_norms": mean_norm,
        "std_of_class_mean_norms": std_norm,
        "distance_of_mean_class_means_origin":centered ,
        "max_diff": max_diff,
        "cond": torch.linalg.cond(sigma_w_inv_b).item(),
        "cond_w": torch.linalg.cond(sigma_w).item(),
        "cond_b": torch.linalg.cond(sigma_b).item()
    }

    return metrics
    
def spectral_entropy(matrix, eps=1e-10):
    evals = torch.linalg.eigvalsh(matrix.to(torch.float32))  # real eigenvalues for symmetric/PSD
    evals = torch.clamp(evals, min=eps)
    evals_norm = evals / evals.sum()
    return -(evals_norm * evals_norm.log()).sum().item()



# def subspace_condition_number(sigma_b: torch.Tensor, n: int = 128, eps: float = 1e-12) -> float:
#     """
#     Computes the condition number of the covariance matrix sigma_b restricted to its
#     non-zero eigenspace (up to rank n - 1 if sample size n is given).

#     Args:
#         sigma_b (torch.Tensor): d x d covariance matrix.
#         n (int, optional): Sample size used to estimate sigma_b. If given, will use
#                            at most n-1 eigenvalues (due to centering).
#         eps (float): Small value to avoid divide-by-zero or log(0).

#     Returns:
#         float: Condition number (λ_max / λ_min) of the non-zero eigenspace.
#                Returns float('inf') if not enough non-zero eigenvalues.
#     """
#     eigvals = torch.linalg.eigvalsh(sigma_b)

#     # Keep only eigenvalues greater than eps
#     eigvals = eigvals[eigvals > eps]

#     # If sample size is given, only consider top (n - 1) eigenvalues
#     if n is not None and eigvals.numel() > (n - 1):
#         eigvals = eigvals[-(n - 1):]

#     if eigvals.numel() < 2:
#         return float('inf')  # Not enough directions to compute condition number

#     λ_max = eigvals.max()
#     λ_min = eigvals.min()
#     return (λ_max / (λ_min + eps)).item()


