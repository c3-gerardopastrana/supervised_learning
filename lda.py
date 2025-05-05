

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from sklearn.covariance import ledoit_wolf_shrinkage




def soft_thresholding(X, lmbda):
    return torch.sign(X) * torch.clamp(torch.abs(X) - lmbda, min=0.0)

def graphical_lasso(S, lmbda=1, max_iter=5, tol=1e-4):
    """
    Sparse inverse covariance estimation via graphical lasso (block coordinate descent).
    
    Args:
        S (torch.Tensor): Empirical covariance matrix, shape (p, p)
        lmbda (float): Sparsity regularization strength (λ)
        max_iter (int): Max iterations
        tol (float): Convergence tolerance
    
    Returns:
        Theta (torch.Tensor): Estimated sparse inverse covariance (precision matrix)
        Sigma (torch.Tensor): Estimated covariance matrix (inverse of Theta)
    """
    print('start')
    p = S.shape[0]
    W = S.clone()
    Theta = torch.inverse(W)
    
    for iteration in range(max_iter):
        Theta_old = Theta.clone()

        for j in range(p):
            # Partition the matrix
            idx = [i for i in range(p) if i != j]
            S11 = S[idx][:, idx]
            s12 = S[idx, j]
            theta12 = Theta[idx, j]

            # Solve lasso subproblem: β = argmin ½ βᵀ S11 β - s12ᵀ β + λ‖β‖₁
            # Using coordinate descent on β
            beta = theta12.clone()
            for _ in range(10):  # inner loop for convergence
                for k in range(p - 1):
                    tmp = s12[k] - S11[k, :].dot(beta) + S11[k, k] * beta[k]
                    beta[k] = soft_thresholding(tmp, lmbda) / S11[k, k]
            
            # Update precision matrix
            Theta[j, idx] = Theta[idx, j] = -beta
            Theta[j, j] = 1.0 / (S[j, j] - s12.dot(beta))

        # Check convergence
        delta = torch.norm(Theta - Theta_old, p='fro') / torch.norm(Theta_old, p='fro')
        if delta < tol:
            break

    #Sigma = torch.inverse(Theta)
    return Theta



def lda(X, y, n_classes, lamb):
    X = X.view(X.shape[0], -1)
    N, D = X.shape
    
    # Find unique classes present in the data
    labels, counts = torch.unique(y, return_counts=True)
    
    # Calculate overall mean and centered data
    X_bar = X - torch.mean(X, 0)
    
    # Calculate total scatter matrix
    St = X_bar.t().matmul(X_bar) / (N - 1)
    
    # Initialize within-class scatter matrix with requires_grad=True
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device, requires_grad=True)
    
    # Pre-allocate tensor for class means
    Xc_mean = torch.zeros((n_classes, D), dtype=X.dtype, device=X.device)
    
    # Calculate within-class scatter matrix
    for c, Nc in zip(labels, counts):
        Xc = X[y == c]
        # Store class mean in the pre-allocated tensor at index c
        Xc_mean[int(c), :] = torch.mean(Xc, 0)
        
        # Center the data for this class
        Xc_bar = Xc - Xc_mean[int(c), :]
        
        # Add weighted contribution to within-class scatter
        # Use max(1, Nc-1) to avoid division by zero for single-sample classes
        Sw = Sw + Xc_bar.t().matmul(Xc_bar) / max(1, Nc - 1) * (Nc / N)
    
    # Calculate between-class scatter matrix
    Sb = St - Sw
    
    # Add regularization to Sw
    #lamb = torch.trace(Sw) / D
    #Sw = graphical_lasso(Sw)
    
    Sw = Sw + torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * lamb
    #Sw_np = Sw.cpu().detach().numpy()
    #lw = ledoit_wolf().fit(Sw_np)  # add a sample axis if needed
    # shrinkage = 0.8
    # mu = torch.trace(Sw) / D

    # Apply shrinkage — differentiable
    # Sw = (1 - shrinkage) * Sw + shrinkage * mu * torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False)
    #Sw =  Sw +  mu * torch.eye(Sw.shape[0], device=Sw.device, dtype=Sw.dtype)
    #Sw =  mu * torch.eye(Sw.shape[0], device=Sw.device, dtype=Sw.dtype)
    # diag_values = torch.rand(D, device="cuda") * 0.0001  # Uniform[0, lam)
    # Sw = Sw + torch.diag(diag_values)
    
    # L = torch.linalg.cholesky(Sw)
    # I = torch.eye(L.size(0), device=L.device, dtype=L.dtype)
    # L_inv = torch.linalg.solve(L, I)  # L^{-1}
    # temp = L_inv @ Sb @ L_inv.T  # This is similar to Sw^{-1/2} @ Sb @ Sw^{-1/2}
    
    # L = torch.linalg.cholesky(Sw)
    # A1 = torch.linalg.solve_triangular(L, Sb, upper=False)
    # temp = torch.linalg.solve_triangular(L.T, Sb, upper=True, left=False)

    
    temp =  torch.linalg.lstsq(Sw, Sb).solution #torch.linalg.solve(Sw, Sb) # # #torch.linalg.pinv(Sw, hermitian=True).matmul(Sb) 
    #temp = 0.5 * (temp + temp.T) 

    # # evals, evecs = torch.symeig(temp, eigenvectors=True) # only works for symmetric matrix
    # evals, evecs = torch.eig(temp, eigenvectors=True) # shipped from nightly-built version (1.8.0.dev20201015)
    # print(evals.shape, evecs.shape)

    # # remove complex eigen values and sort
    # noncomplex_idx = evals[:, 1] == 0
    # evals = evals[:, 0][noncomplex_idx] # take real part of eigen values
    # evecs = evecs[:, noncomplex_idx]
    # evals, inc_idx = torch.sort(evals) # sort by eigen values, in ascending order
    # evecs = evecs[:, inc_idx]
    # print(evals.shape, evecs.shape)

    # # flag to indicate if to skip backpropagation
    # hasComplexEVal = evecs.shape[1] < evecs.shape[0]

    # return hasComplexEVal, Xc_mean, evals, evecs
    # compute eigen decomposition
    # evals, evecs = torch.symeig(temp, eigenvectors=True) # only works for symmetric matrix
    # Use the new torch.linalg.eig for general matrices
    # It returns complex eigenvalues and eigenvectors by default
    #temp = temp.to(dtype=torch.float32)


    
    evals_complex, evecs_complex = torch.linalg.eig(temp)
    

    # Process complex eigenvalues returned by torch.linalg.eig
    # Check for eigenvalues with non-negligible imaginary parts
    tol = 1e-6 # Tolerance for considering imaginary part zero
    is_complex = torch.abs(evals_complex.imag) > tol
    hasComplexEVal = torch.any(is_complex) # Flag if *any* eigenvalue was complex beyond tolerance

    if hasComplexEVal:
         # Optional: Print a warning if complex eigenvalues are detected
         print(f"Warning: Found {torch.sum(is_complex)} eigenvalues with imaginary part > {tol}. Keeping only real eigenvalues.")

    real_idx = ~is_complex
    evals = evals_complex[real_idx].real 
    evecs = evecs_complex[:, real_idx].real

    if evals.numel() > 0: # Check if any real eigenvalues are left
        evals, inc_idx = torch.sort(evals)
        evecs = evecs[:, inc_idx]
    else:
        print("Warning: All eigenvalues were complex. Eigenvalue/vector tensors are empty.")
        evals = torch.tensor([], dtype=temp.dtype, device=temp.device)
        D = temp.shape[0]
        evecs = torch.tensor([[] for _ in range(D)], dtype=temp.dtype, device=temp.device)
    return hasComplexEVal, Xc_mean, evals, evecs, temp


def lda_loss(evals, n_classes, n_eig=None, margin=None):
    n_components = n_classes - 1
    evals = evals[-n_components:]
    # evecs = evecs[:, -n_components:]
    # print('evals', evals.shape, evals)
    # print('evecs', evecs.shape)

    # calculate loss
    if margin is not None:
        threshold = torch.min(evals) + margin
        n_eig = torch.sum(evals < threshold)
   
    probs = evals / evals.sum()
    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-9)))

    loss = -entropy
    #loss = torch.mean(evals[:n_eig]) # small eigen values are on left
    
    # eigvals_norm = evals / evals.sum()
    # eps = 1e-10 
    # eigvals_norm = torch.clamp(eigvals_norm, min=eps)
    # oss = (eigvals_norm * eigvals_norm.log()).sum()
    #loss = torch.log(eigvals_norm.max()-eigvals_norm.min())
    return loss
    
def sina_loss(sigma_w_inv_b):
    n = torch.tensor(512, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)

    max_frobenius_norm = max_frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
    max_frobenius_norm = torch.sqrt(max_frobenius_norm.abs()) 
    
    trace = torch.trace(sigma_w_inv_b).abs()



    # lambda_target = torch.tensor(512, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    

    # # dim = sigma_w_inv_b.shape[0]
    # # identity = torch.eye(dim, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    # # diff = sigma_w_inv_b - lambda_target * identity
    # # loss = torch.norm(diff, p='fro')**2

    # penalty = (trace - lambda_target).pow(2)  # scale-free, minimal tuning
    # lambda_target = torch.tensor(2**10, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    # penalty = (trace - lambda_target).pow(2) / lambda_target  # scale-free, minimal tuning
    lambda_target = torch.tensor(2**10, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    diff = lambda_target - trace  # positive if trace < lambda_target
    penalty = F.relu(diff).pow(2) / lambda_target.pow(2)  # scale-free, no penalty if trace >= lambda_target

    loss = torch.log(max_frobenius_norm) -   torch.log(trace) + penalty
    
    # sigma_b_inv_w =  torch.linalg.pinv(sigma_w_inv_b, hermitian=True)
    # min_frobenius_norm = torch.trace(sigma_b_inv_w @ sigma_b_inv_w)
    # min_frobenius_norm = 1/torch.sqrt(min_frobenius_norm.abs())
    
    # gap =  max_frobenius_norm - min_frobenius_norm
    #loss = torch.log(gap / trace)
    
    #trace = torch.trace(1/2*(sigma_w_inv_b + sigma_w_inv_b.T))
    #loss = torch.log(max_frobenius_norm) - torch.log(trace)
    
    # off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
    # sum_squared_off_diag = torch.sum(off_diag ** 2).item()
    # loss = sum_squared_off_diag - trace
    return loss

class LDA(nn.Module):
    def __init__(self, n_classes, lamb):
        super(LDA, self).__init__()
        self.n_classes = n_classes
        self.n_components = n_classes - 1
        self.lamb = lamb
        self.lda_layer = partial(lda, n_classes=n_classes, lamb=lamb)

    def forward(self, X, y):
        # Perform batch-wise LDA (temporary, not global yet)
        hasComplexEVal, Xc_mean, evals, evecs, sigma_w_inv_b = self.lda_layer(X, y)

        # Save batch-wise scalings (not necessarily global yet)
        self.scalings_ = evecs
        self.coef_ = Xc_mean.matmul(evecs).matmul(evecs.t())
        self.intercept_ = -0.5 * torch.diagonal(Xc_mean.matmul(self.coef_.t()))

        return hasComplexEVal, evals, sigma_w_inv_b

    def transform(self, X):
        return X.matmul(self.scalings_)[:, :self.n_components]

    def predict(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return torch.argmax(logit, dim=1)

    def predict_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return nn.functional.softmax(logit, dim=1)

    def predict_log_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return nn.functional.log_softmax(logit, dim=1)



class RunningLDAStats:
    def __init__(self, n_classes, n_features, device='cpu'):
        self.n_classes = n_classes
        self.n_features = n_features
        self.device = 'cpu'  # FORCE CPU

        self.class_sums = torch.zeros((n_classes, n_features), device=self.device)
        self.class_counts = torch.zeros(n_classes, device=self.device)
        self.within_class_scatter = torch.zeros((n_features, n_features), device=self.device)

    @torch.no_grad()
    def update(self, X, y):
        X = X.view(X.shape[0], -1).detach().to('cpu')
        y = y.detach().to('cpu')

        for cls in range(self.n_classes):
            mask = (y == cls)
            if mask.sum() == 0:
                continue
            Xc = X[mask]
            Nc = Xc.shape[0]
            mean_c = Xc.mean(dim=0)

            self.class_sums[cls] += Xc.sum(dim=0)
            self.class_counts[cls] += Nc

            Xc_centered = Xc - mean_c
            scatter_c = Xc_centered.t().matmul(Xc_centered)
            self.within_class_scatter += scatter_c


    def finalize(self, lamb=1e-4):
        means = self.class_sums / self.class_counts.unsqueeze(1)

        total_samples = self.class_counts.sum()
        overall_mean = self.class_sums.sum(dim=0) / total_samples

        Sb = torch.zeros((self.n_features, self.n_features), device=self.device)
        for cls in range(self.n_classes):
            Nc = self.class_counts[cls]
            if Nc > 0:
                mean_diff = (means[cls] - overall_mean).unsqueeze(1)
                Sb += Nc * mean_diff.matmul(mean_diff.t())

        Sw = self.within_class_scatter + lamb * torch.eye(self.n_features, device=self.device)

        return Sw, Sb, means


def spherical_lda(X, y, n_classes, lamb):
    N, D = X.shape
    labels, counts = torch.unique(y, return_counts=True)
    assert len(labels) == n_classes  # require all classes to be present
    
    # Compute global mean direction and normalize
    global_mean = torch.mean(X, 0)
    global_mean = F.normalize(global_mean, p=2, dim=0)
    
    # Initialize containers
    class_means_list = []
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device)
    Sb = torch.zeros((D, D), dtype=X.dtype, device=X.device)
    
    # Calculate all class means
    for c in labels:
        class_idx = int(c)
        Xc = X[y == c]
        class_mean = F.normalize(torch.mean(Xc, dim=0), p=2, dim=0)
        class_means_list.append(class_mean)
    
    Xc_mean = torch.stack(class_means_list)
    
    # Compute scatter matrices
    for i, (c, Nc) in enumerate(zip(labels, counts)):
        Xc = X[y == c]
        class_mean = Xc_mean[i]

        # Vectorized cosine similarities
        cos_similarities = Xc @ class_mean
        cos_similarities = torch.clamp(cos_similarities, -1.0 + 1e-6, 1.0 - 1e-6)

        # Vectorized difference: samples projected away from mean direction
        diffs = Xc - cos_similarities.unsqueeze(1) * class_mean.unsqueeze(0)

        # Vectorized scatter matrix
        class_scatter = diffs.T @ diffs

        Sw = Sw + class_scatter
    
    Sw = Sw / N  # Normalize by total number of samples
    
    # Compute between-class scatter
    for i, (c, Nc) in enumerate(zip(labels, counts)):
        class_mean = Xc_mean[i]
        cos_sim = torch.dot(class_mean, global_mean)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)

        diff = class_mean.unsqueeze(1) - cos_sim * global_mean.unsqueeze(1)
        diff_outer = diff @ diff.T
        Sb = Sb + (Nc / N) * diff_outer

    
    # import numpy as np
    # from sklearn.covariance import LedoitWolf
    
    # X_np = Sw.cpu().detach().numpy()

    # lw = LedoitWolf()
    # lw.fit(X_np)
    # shrinkage = lw.shrinkage_
    # mu = torch.trace(Sw) / D
    # shrinkage = torch.nn.Parameter(torch.tensor(0.0, dtype=Sw.dtype, device=Sw.device))
    # shrinkage = torch.sigmoid(shrinkage)
    
    Sw_reg = Sw + torch.eye(D, dtype=X.dtype, device=X.device) * lamb
    #Sw_reg = (1-shrinkage) * Sw + torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * shrinkage * mu
    
    
    # Generalized eigenvalue problem
    temp = torch.linalg.pinv(Sw_reg, hermitian=True) @ Sb
    
    # Eigen decomposition
    evals_complex, evecs_complex = torch.linalg.eig(temp)
    tol = 1e-6
    is_complex = torch.abs(evals_complex.imag) > tol
    hasComplexEVal = torch.any(is_complex)
    
    if hasComplexEVal:
        print(f"Warning: Found {torch.sum(is_complex)} eigenvalues with imaginary part > {tol}. Keeping only real ones.")
    
    real_idx = ~is_complex
    evals = evals_complex[real_idx].real
    evecs = evecs_complex[:, real_idx].real
    
    if evals.numel() > 0:
        evals, inc_idx = torch.sort(evals)
        evecs = evecs[:, inc_idx]
    else:
        print("Warning: All eigenvalues were complex.")
        evals = torch.tensor([], dtype=temp.dtype, device=temp.device)
        evecs = torch.zeros((D, 0), dtype=temp.dtype, device=temp.device)
    
    return hasComplexEVal, Xc_mean, evals, evecs, temp


class SphericalLDA(nn.Module):
    def __init__(self, n_classes, lamb=1e-4):
        super(SphericalLDA, self).__init__()
        self.n_classes = n_classes
        self.n_components = n_classes - 1  # Maximum meaningful LDA dimensions
        self.lamb = lamb
        self.lda_layer = partial(spherical_lda, n_classes=n_classes, lamb=lamb)
    
    def forward(self, X, y):
        hasComplexEVal, Xc_mean, evals, evecs, sigma_w_inv_b = self.lda_layer(X, y)
        
        # Store projection matrix 
        self.scalings_ = evecs
        
        # Project class means and normalize to create prototypes
        projected_means = Xc_mean.matmul(evecs)
        
        # Project back to original space and normalize to ensure they're on the hypersphere
        self.coef_ = F.normalize(projected_means.matmul(evecs.t()), p=2, dim=1)
        
        # Intercept is not meaningful in spherical space when using cosine similarity
        self.intercept_ = torch.zeros(self.n_classes, dtype=X.dtype, device=X.device)
        
        return hasComplexEVal, evals, sigma_w_inv_b
    
    def transform(self, X):
        # Normalize input
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        
        # Project data
        X_new = X.matmul(self.scalings_)
        
        # Return only the most discriminative components
        return X_new[:, :self.n_components]
    
    def predict(self, X):
        # Normalize input embeddings
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        
        # Compute cosine similarities with class prototypes
        similarities = X.matmul(self.coef_.t())
        
        # Return class with highest similarity
        return torch.argmax(similarities, dim=1)
    
    def predict_proba(self, X):
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        similarities = X.matmul(self.coef_.t())
        
        # Convert similarities to probabilities using softmax
        proba = nn.functional.softmax(similarities, dim=1)
        return proba
    
    def predict_log_proba(self, X):
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        similarities = X.matmul(self.coef_.t())
        log_proba = nn.functional.log_softmax(similarities, dim=1)
        return log_proba

if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    features, labels = load_iris(return_X_y=True)
    print(features.shape, labels.shape)

    n_classes = 3
    n_components = n_classes - 1
    N, D = features.shape  # 150, 4
    lamb = 0.001
    n_eig = 2
    margin = 0.01

    device = torch.device('cpu:0')
    X = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)

    lda = LDA(n_classes, lamb)
    _, evals = lda(X, y)

    # calculate lda loss
    loss = lda_loss(evals, n_classes, n_eig, margin)
    loss.backward()
    print('finished backward')

    # use LDA as classifier
    y_pred = lda.predict(X)
    print('accuracy on training data', accuracy_score(y, y_pred))