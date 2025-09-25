### Script for Baselines

# Standard library
import itertools

# Core libraries
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
import torch_geometric.utils

# HuggingFace Transformers
from transformers import AutoModel, AutoTokenizer

# Scikit-learn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    pairwise_distances,
    r2_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Constrained clustering
from k_means_constrained import KMeansConstrained

# Vision
import torchvision
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights

##############################################
# FixMatch #
##############################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class FixMatchRegressor:
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=1e-3,
        alpha_ema=0.99,
        lambda_u_max=1.0,
        rampup_length=10,
        conf_threshold=0.1,   # threshold on std of pseudo-labels
        device=None
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Student and EMA teacher
        self.student = MLPRegressor(input_dim, output_dim).to(device)
        self.teacher = MLPRegressor(input_dim, output_dim).to(device)
        self._update_teacher(ema_decay=0.0)  # initialize teacher = student

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.ema_decay = alpha_ema

        # Unsupervised weight schedule
        self.lambda_u_max = lambda_u_max
        self.rampup_length = rampup_length

        # Confidence threshold (we’ll measure std of multiple weak preds)
        self.conf_threshold = conf_threshold

        # MSE criterion
        self.mse_loss = nn.MSELoss(reduction="none")

    @torch.no_grad()
    def _update_teacher(self, ema_decay=None):
        """EMA update: teacher_params = ema_decay * teacher + (1-ema_decay) * student"""
        decay = self.ema_decay if ema_decay is None else ema_decay
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=(1.0 - decay))

    def _get_lambda_u(self, current_epoch):
        """Linear ramp-up from 0 -> lambda_u_max over rampup_length epochs"""
        if current_epoch >= self.rampup_length:
            return self.lambda_u_max
        else:
            return self.lambda_u_max * (current_epoch / self.rampup_length)

    def _augment(self, x, strong=False):
        """
        Return an augmented version of x.
        - strong: heavier noise + random dimension dropout.
        """
        # 1) Additive Gaussian noise (relative to feature scale)
        noise_scale = 0.05 if not strong else 0.2
        x_noisy = x + torch.randn_like(x) * noise_scale

        # 2) Multiplicative jitter (small random scale per-dimension)
        if strong:
            scale = 1.0 + 0.1 * torch.randn_like(x)
            x_noisy = x_noisy * scale

        # 3) Random dimension dropout (only for strong)
        if strong:
            # randomly zero out 10% of dimensions
            mask = (torch.rand_like(x) > 0.1).float()
            x_noisy = x_noisy * mask

        return x_noisy

    def train(
        self,
        sup_loader: DataLoader,
        unl_loader: DataLoader,
        epochs: int = 100
    ):
        """
        Train student+teacher for `epochs` epochs.
        sup_loader yields (x_sup, y_sup)
        unl_loader yields (x_unl, dummy)  [we ignore dummy]
        """
        self.student.train()
        self.teacher.train()

        min_batches = min(len(sup_loader), len(unl_loader))

        for epoch in range(epochs):
            lambda_u = self._get_lambda_u(epoch)
            epoch_losses = {"sup": 0.0, "unsup": 0.0}

            sup_iter = iter(sup_loader)
            unl_iter = iter(unl_loader)

            for _ in range(min_batches):
                # --- 1) Fetch one sup batch and one unl batch
                x_sup, y_sup = next(sup_iter)
                x_unl, _ = next(unl_iter)

                x_sup = x_sup.float().to(self.device)
                y_sup = y_sup.float().to(self.device)
                x_unl = x_unl.float().to(self.device)

                # --- 2) Supervised forward
                preds_sup = self.student(x_sup)                    # (B, out_dim)
                loss_sup = F.mse_loss(preds_sup, y_sup, reduction="mean")

                # --- 3) Unlabeled: generate multiple weak views for confidence
                # We’ll do two weak augmentations per sample.
                x_unl_w1 = self._augment(x_unl, strong=False)
                x_unl_w2 = self._augment(x_unl, strong=False)

                with torch.no_grad():
                    # Teacher produces pseudo-labels
                    p_w1 = self.teacher(x_unl_w1)  # (B, out_dim)
                    p_w2 = self.teacher(x_unl_w2)  # (B, out_dim)

                    # Estimate “confidence” by the two weak preds’ difference
                    pseudo_label = 0.5 * (p_w1 + p_w2)  # average as final pseudo
                    std_weak = (p_w1 - p_w2).pow(2).mean(dim=1).sqrt()  # (B,) L2‐std

                # Mask = 1 if std_weak < threshold, else 0
                mask = (std_weak < self.conf_threshold).float().unsqueeze(1)  # (B,1)

                # --- 4) Strong aug on unlabeled
                x_unl_s = self._augment(x_unl, strong=True)
                preds_s = self.student(x_unl_s)  # (B, out_dim)

                # --- 5) Unsupervised consistency loss (only for “confident” samples)
                # We compute MSE per-sample, then multiply by mask
                loss_unsup_per_sample = self.mse_loss(preds_s, pseudo_label)  # (B, out_dim)
                # average over output_dim, then multiply by mask
                loss_unsup = (loss_unsup_per_sample.mean(dim=1) * mask.squeeze(1)).mean()

                # --- 6) Total loss
                loss = loss_sup + lambda_u * loss_unsup

                # --- 7) Backprop & update student
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # --- 8) EMA update for teacher
                self._update_teacher()

                epoch_losses["sup"] += loss_sup.item()
                epoch_losses["unsup"] += loss_unsup.item()


    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        """Use the EMA teacher to predict on new data."""
        self.teacher.eval()
        X = X.float().to(self.device)
        return self.teacher(X).cpu().numpy()

def fixmatch_regression(
    supervised_df,
    input_only_df,
    test_df,
    epochs=5000,
    batch_size=32,
    lr=1e-3,
    alpha_ema=0.99,
    lambda_u_max=1.0,
    rampup_length=10,
    conf_threshold=0.1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Prepare numpy arrays
    X_sup = np.vstack(supervised_df["morph_coordinates"])
    y_sup = np.vstack(supervised_df["gene_coordinates"])
    X_unl = np.vstack(input_only_df["morph_coordinates"])   # unlabeled pool (inputs only)
    X_tst = np.vstack(test_df["morph_coordinates"])         # held-out test
    y_tst = np.vstack(test_df["gene_coordinates"])

    input_dim = X_sup.shape[1]
    output_dim = y_sup.shape[1]

    # 2) Build datasets & loaders
    sup_dataset = TensorDataset(
        torch.tensor(X_sup, dtype=torch.float32),
        torch.tensor(y_sup, dtype=torch.float32)
    )
    unl_dataset = TensorDataset(
        torch.tensor(X_unl, dtype=torch.float32),
        torch.zeros((len(X_unl), y_sup.shape[1]), dtype=torch.float32)
    )
    sup_loader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    unl_loader = DataLoader(unl_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3) Initialize FixMatchRegressor
    fixmatch = FixMatchRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        lr=lr,
        alpha_ema=alpha_ema,
        lambda_u_max=lambda_u_max,
        rampup_length=rampup_length,
        conf_threshold=conf_threshold,
        device=device
    )

    # 4) Train
    fixmatch.train(sup_loader, unl_loader, epochs=epochs)

    # 5) Final inference on unlabeled
    X_tst_tensor = torch.tensor(X_tst, dtype=torch.float32)
    preds_tst = fixmatch.predict(X_tst_tensor)

    return preds_tst, y_tst

############################################################
# Laplacian‑Regularised Least‑Squares (LapRLS)
############################################################

import numpy as np
from sklearn.metrics import pairwise_distances

def laprls_closed_form(Xs, ys, Xu, lam=1e-2, gamma=1.0, k=10, sigma=None):
    """
    Laplacian-Regularized Least Squares (linear model)

    Solves:
       w = argmin_w  ||Xs w - ys||^2  +  lam ||w||^2  +  gamma * w^T (X^T L X) w

    where L is the graph Laplacian on the concatenated data [Xs; Xu].
    Note: in Belkin et al., the Laplacian term is gamma_I/(l+u)^2 * f^T L f;
    here we absorb 1/(l+u)^2 into `gamma`.

    Params
    ------
    Xs : array (l × d)    labeled inputs
    ys : array (l × m)    labeled targets
    Xu : array (u × d)    unlabeled inputs
    lam: float            Tikhonov weight λ
    gamma: float          Laplacian weight γ (already includes any 1/(l+u)^2)
    k: int                number of nearest neighbors
    sigma: float or None  RBF kernel width (if None, set to median pairwise distance)

    Returns
    -------
    w : array (d × m)      regression weights
    """
    # Stack all inputs
    X = np.vstack([Xs, Xu])
    n = X.shape[0]

    # Estimate sigma if needed
    if sigma is None:
        # median of pairwise Euclidean distances
        dists = pairwise_distances(X, metric='euclidean')
        sigma = np.median(dists[dists>0])

    # Build adjacency with RBF similarities
    gamma_rbf = 1.0 / (2 * sigma**2)
    S = np.exp(- pairwise_distances(X, X, squared=True) * gamma_rbf)

    # kNN sparsification
    idx = np.argsort(-S, axis=1)[:, 1:k+1]
    W = np.zeros_like(S)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    W[rows, cols] = S[rows, cols]
    W = np.maximum(W, W.T)  # symmetrize

    # Normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
    deg = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-12))
    L = np.eye(n) - D_inv_sqrt.dot(W).dot(D_inv_sqrt)

    # Closed-form solve
    # A = Xs^T Xs + lam I + gamma * X^T L X
    A = Xs.T.dot(Xs) + lam*np.eye(X.shape[1]) + gamma * X.T.dot(L).dot(X)
    B = Xs.T.dot(ys)
    w = np.linalg.solve(A, B)
    return w

def laprls_regression(supervised_df, input_only_df, test_df, lam=1e-2, gamma=1.0, k=10, sigma=None):
    Xs = np.vstack(supervised_df['morph_coordinates'])
    ys = np.vstack(supervised_df['gene_coordinates'])
    Xu = np.vstack(input_only_df['morph_coordinates'])   # unlabeled inputs for Laplacian
    Xt = np.vstack(test_df['morph_coordinates'])         # held-out test inputs
    yt = np.vstack(test_df['gene_coordinates'])
    w = laprls_closed_form(Xs, ys, Xu, lam, gamma, k, sigma)
    preds = Xt.dot(w)
    return preds, yt


############################################################
# Twin‑Neural‑Network Regression (TNNR)
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools

# --- Supervised pairwise differences ---------------------------
class PairwiseDataset(Dataset):
    """
    Supervised dataset of all (i, j) pairs from (X, y),
    with targets y_i - y_j.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.pairs = list(itertools.combinations(range(len(X)), 2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        xi, xj = self.X[i], self.X[j]
        dy = self.y[i] - self.y[j]
        return xi, xj, dy

# --- Unsupervised loop‐consistency triples ----------------------
class LoopConsistencyDataset(Dataset):
    """
    Unlabeled dataset of random triples (i, j, k) from X,
    for enforcing f(x_i,x_j) + f(x_j,x_k) + f(x_k,x_i) ≈ 0.
    """
    def __init__(self, X, n_loops=5):
        self.X = torch.tensor(X, dtype=torch.float32)
        n = len(X)
        # generate n_loops * n random triples
        self.triples = [
            tuple(np.random.choice(n, 3, replace=False))
            for _ in range(n_loops * n)
        ]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        i, j, k = self.triples[idx]
        return self.X[i], self.X[j], self.X[k]

# --- Twin-Neural-Network Regression Model -----------------------
class TwinRegressor(nn.Module):
    """
    Shared encoder h, difference head g:
      f(x1, x2) = g(h(x1) - h(x2))
    """
    def __init__(self, in_dim, rep_dim=64, out_dim=1):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rep_dim),
            nn.ReLU()
        )
        self.g = nn.Linear(rep_dim, out_dim)

    def forward(self, x1, x2):
        h1 = self.h(x1)
        h2 = self.h(x2)
        return self.g(h1 - h2)

def tnnr_regression(
    sup_df,
    input_only_df,
    test_df,
    rep_dim=64,
    beta=0.1,           # loop-consistency weight
    lr=1e-3,
    epochs=5000,
    batch_size=256,
    n_loops=2,
    device=None
):
    """
    Twin-NN regression with loop consistency (Sec. 3.2, TNNR paper).
    Trains on supervised pairwise differences + unlabeled loops.
    """
    # Prepare data arrays
    Xs = np.vstack(sup_df['morph_coordinates']);    ys = np.vstack(sup_df['gene_coordinates'])
    Xu = np.vstack(input_only_df['morph_coordinates'])     # unlabeled for loops
    Xq = np.vstack(test_df['morph_coordinates'])           # held-out queries
    yq = np.vstack(test_df['gene_coordinates'])

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets and loaders
    pair_ds = PairwiseDataset(Xs, ys)
    loop_ds = LoopConsistencyDataset(Xu, n_loops=n_loops)
    pair_loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    loop_loader = DataLoader(loop_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Model, optimizer, loss
    in_dim  = Xs.shape[1]
    out_dim = ys.shape[1]
    model = TwinRegressor(in_dim, rep_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        for (x1, x2, dy), (xu1, xu2, xu3) in zip(pair_loader, loop_loader):
            x1, x2, dy = x1.to(device), x2.to(device), dy.to(device)
            xu1, xu2, xu3 = xu1.to(device), xu2.to(device), xu3.to(device)

            # Supervised pairwise loss
            pred_sup = model(x1, x2)
            loss_sup = mse(pred_sup, dy)

            # Loop‐consistency loss
            lo_ij = model(xu1, xu2)
            lo_jk = model(xu2, xu3)
            lo_ki = model(xu3, xu1)
            loss_loop = mse(lo_ij + lo_jk + lo_ki, torch.zeros_like(lo_ij))

            # Combine and optimize
            loss = loss_sup + beta * loss_loop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Inference: ensemble differences to all supervised anchors
    model.eval()
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    ys_t = torch.tensor(ys, dtype=torch.float32, device=device)
    Xq_t = torch.tensor(Xq, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for xq in Xq_t:
            diffs = model(xq.unsqueeze(0).repeat(len(Xs_t), 1), Xs_t)
            estimates = ys_t + diffs
            preds.append(estimates.mean(dim=0).cpu().numpy())
    return np.vstack(preds), yq


##############################################
#  Transductive SVM‑Regression (TSVR)
############################################## 

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def tsvr_regression(
    supervised_df,
    input_only_df,
    test_df,
    C=1.0,
    epsilon=0.1,
    kernel='rbf',
    gamma='scale',
    self_training_frac=0.2
):
    # 1) Prepare data
    X_sup = np.vstack(supervised_df['morph_coordinates'])
    y_sup = np.vstack(supervised_df['gene_coordinates'])
    X_unl = np.vstack(input_only_df['morph_coordinates'])   # unlabeled pool for self-training
    X_tst = np.vstack(test_df['morph_coordinates'])         # held-out test
    y_tst = np.vstack(test_df['gene_coordinates'])

    # 2) Base SVR wrapped for multi-output
    base_svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
    model = MultiOutputRegressor(base_svr)

    # 3) Initial fit on supervised only
    model.fit(X_sup, y_sup)
    pseudo = model.predict(X_unl)

    # TSVR is the slowest of the baselines, so we limit max_iter to a reasonable value, drawing fair comparisons
    reasonable_max_iter = 5

    for it in range(reasonable_max_iter):
        if it == 0:
            # first self-training round: include all unlabeled
            X_aug = np.vstack([X_sup, X_unl])
            y_aug = np.vstack([y_sup, pseudo])
        else:
            # measure stability of each pseudolabel
            diffs = np.linalg.norm(pseudo - prev_pseudo, axis=1)
            # pick the fraction with smallest change
            thresh = np.percentile(diffs, self_training_frac * 100)
            mask = diffs <= thresh
            X_aug = np.vstack([X_sup, X_unl[mask]])
            y_aug = np.vstack([y_sup, pseudo[mask]])

        # 4) Refit on augmented data
        model.fit(X_aug, y_aug)

        # 5) Check convergence
        prev_pseudo = pseudo
        pseudo = model.predict(X_unl)
        if np.allclose(pseudo, prev_pseudo, atol=1e-3):
            break

    preds = model.predict(X_tst)
    # final predictions on unlabeled
    return preds, y_tst


##############################################
# Uncertainty‑Consistent VME (UCVME) 
##############################################
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UCVMEModel(nn.Module):
    """
    A Bayesian-style regressor that jointly predicts a mean and a log-variance.
    """
    def __init__(self, in_dim, out_dim, hidden=128, p_dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(p_dropout),
        )
        self.mean_head   = nn.Linear(hidden, out_dim)
        self.logvar_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.encoder(x)
        return self.mean_head(h), self.logvar_head(h)


def ucvme_regression(
    supervised_df,
    input_only_df,
    test_df,
    mc_T=5,                  # MC dropout samples
    lr=1e-3,
    epochs=50,           
    w_unl=10.0,          
    device=None
):
    T = mc_T  # number of MC dropout samples
    # — Prepare tensors —
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_sup = torch.tensor(np.vstack(supervised_df['morph_coordinates']), dtype=torch.float32, device=device)
    y_sup = torch.tensor(np.vstack(supervised_df['gene_coordinates']),    dtype=torch.float32, device=device)
    X_unl = torch.tensor(np.vstack(input_only_df['morph_coordinates']),   dtype=torch.float32, device=device)  # unlabeled pool
    X_tst = torch.tensor(np.vstack(test_df['morph_coordinates']),         dtype=torch.float32, device=device)
    y_tst_np = np.vstack(test_df['gene_coordinates'])

    in_dim, out_dim = X_sup.shape[1], y_sup.shape[1]
    model_a = UCVMEModel(in_dim, out_dim).to(device)
    model_b = UCVMEModel(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model_a.parameters()) + list(model_b.parameters()),
        lr=lr
    )

    for ep in range(epochs):
        model_a.train(); model_b.train()

        y_a_sup, z_a_sup = model_a(X_sup)
        y_b_sup, z_b_sup = model_b(X_sup)
        L_sup_reg = (
            ((y_a_sup - y_sup)**2 / (2*torch.exp(z_a_sup)) + z_a_sup/2).mean()
          + ((y_b_sup - y_sup)**2 / (2*torch.exp(z_b_sup)) + z_b_sup/2).mean()
        )

        L_sup_unc = ((z_a_sup - z_b_sup)**2).mean()

        y_a_list, z_a_list, y_b_list, z_b_list = [], [], [], []
        for _ in range(T):
            # keep dropout active
            y_a_t, z_a_t = model_a(X_unl)
            y_b_t, z_b_t = model_b(X_unl)
            y_a_list.append(y_a_t); z_a_list.append(z_a_t)
            y_b_list.append(y_b_t); z_b_list.append(z_b_t)

        y_a_stack = torch.stack(y_a_list)  # (T, N_unl, D)
        z_a_stack = torch.stack(z_a_list)
        y_b_stack = torch.stack(y_b_list)
        z_b_stack = torch.stack(z_b_list)

        # average over runs and models
        y_tilde = (y_a_stack.mean(0) + y_b_stack.mean(0)) / 2
        z_tilde = (z_a_stack.mean(0) + z_b_stack.mean(0)) / 2
        L_unl_reg = (
            ((y_a_stack.mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean()
          + ((y_b_stack.mean(0) - y_tilde)**2 / (2*torch.exp(z_tilde)) + z_tilde/2).mean()
        )

        L_unl_unc = (
            ((z_a_stack.mean(0) - z_tilde)**2).mean()
          + ((z_b_stack.mean(0) - z_tilde)**2).mean()
        )

        loss = L_sup_reg + L_sup_unc + w_unl * (L_unl_reg + L_unl_unc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model_a.eval(); model_b.eval()
    preds_list = []
    with torch.no_grad():
        for _ in range(T):
            y_a_u, _ = model_a(X_tst)
            y_b_u, _ = model_b(X_tst)
            preds_list.append((y_a_u + y_b_u) / 2)
        preds = torch.stack(preds_list).mean(0).cpu().numpy()

    return preds, y_tst_np

###################################
#  GCN        #
###################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

# ----- Minimal 2-layer GCN for regression -----
class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ----- Vectorized edge builders -----
def _complete_no_self(n: int, device=None):
    """
    Directed complete graph on {0..n-1} without self-loops.
    Returns edge_index of shape [2, n*(n-1)].
    """
    idx = torch.arange(n, device=device)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)

def _bipartite_both(nA: int, nB: int, offsetA: int, offsetB: int, device=None):
    """
    Directed bipartite edges A<->B.
    A nodes are {offsetA..offsetA+nA-1}, B nodes {offsetB..offsetB+nB-1}.
    Returns edge_index of shape [2, 2*nA*nB].
    """
    A = torch.arange(nA, device=device) + offsetA
    B = torch.arange(nB, device=device) + offsetB
    # A -> B
    src1 = A.repeat_interleave(nB)
    dst1 = B.repeat(nA)
    # B -> A
    src2 = B.repeat_interleave(nA)
    dst2 = A.repeat(nB)
    src = torch.cat([src1, src2], dim=0)
    dst = torch.cat([dst1, dst2], dim=0)
    return torch.stack([src, dst], dim=0)

# ----- Main API -----
def gcn_regression(
    supervised_df,
    input_only_df,
    test_df,
    hidden=64,
    dropout=0.1,
    epochs=100,
    lr=1e-3,
    device=None
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # ======= Assemble splits =======
    sup_m = np.vstack(supervised_df['morph_coordinates'].values)   # (N_sup, in_dim)
    sup_g = np.vstack(supervised_df['gene_coordinates'].values)    # (N_sup, out_dim)
    unl_m = np.vstack(input_only_df['morph_coordinates'].values)   # (N_unl, in_dim)  (no labels used)
    tst_m = np.vstack(test_df['morph_coordinates'].values)         # (N_tst, in_dim)
    tst_g = np.vstack(test_df['gene_coordinates'].values)          # (N_tst, out_dim)

    N_sup = sup_m.shape[0]
    N_unl = unl_m.shape[0]
    N_tst = tst_m.shape[0]

    in_dim  = sup_m.shape[1]
    out_dim = sup_g.shape[1]

    # ======= Build TRAIN graph: [supervised | input_only] =======
    X_train = torch.tensor(np.vstack([sup_m, unl_m]), dtype=torch.float32, device=device)
    Y_sup   = torch.tensor(sup_g, dtype=torch.float32, device=device)

    # edges: complete directed on supervised + bipartite both ways (input_only <-> supervised)
    e_sup_sup = _complete_no_self(N_sup, device=device)  # [2, N_sup*(N_sup-1)]
    e_u_sup   = _bipartite_both(N_unl, N_sup, offsetA=N_sup, offsetB=0, device=device)  # unl <-> sup
    edge_index_train = torch.cat([e_sup_sup, e_u_sup], dim=1)  # [2, E_train]

    # ======= Model / optimizer =======
    model = GCN(in_channels=in_dim, hidden_channels=hidden, out_channels=out_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ======= Training =======
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train, edge_index_train)      # [N_sup + N_unl, out_dim]
        loss = loss_fn(out[:N_sup], Y_sup)          # supervise ONLY on supervised nodes
        loss.backward()
        optimizer.step()

    # ======= Build INFERENCE graph: [supervised | test] =======
    X_infer = torch.tensor(np.vstack([sup_m, tst_m]), dtype=torch.float32, device=device)

    e_sup_sup_inf = _complete_no_self(N_sup, device=device)
    e_t_sup       = _bipartite_both(N_tst, N_sup, offsetA=N_sup, offsetB=0, device=device)  # test <-> sup
    edge_index_infer = torch.cat([e_sup_sup_inf, e_t_sup], dim=1)

    # ======= Predict on test nodes =======
    model.eval()
    with torch.no_grad():
        out_inf = model(X_infer, edge_index_infer)      # [N_sup + N_tst, out_dim]
        preds_tst = out_inf[N_sup:].cpu().numpy()       # only test slice

    return preds_tst, tst_g


###################################
# Unmatched Regression #
####################################

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from adapt.instance_based import KMM  # pip install adapt

def kernel_mean_matching_regression(
    image_df,
    gene_df,
    supervised_df,
    inference_df,
    alpha=1e-2,    # regularization for KRR
    kmm_B=1000,    # B parameter for KMM
    kmm_eps=1e-3,  # eps parameter for KMM
    sigma=None     # optional RBF width; if None it’s auto-computed
):

    # --- 1) Extract raw arrays ---
    X_img   = np.vstack(image_df['morph_coordinates'].values)      # (N_img, d)
    Y_gene  = np.vstack(gene_df['gene_coordinates'].values)        # (N_gene, d)
    X_sup   = np.vstack(supervised_df['morph_coordinates'].values) # (N_sup, d)
    Y_sup   = np.vstack(supervised_df['gene_coordinates'].values)  # (N_sup, d)
    X_test  = np.vstack(inference_df['morph_coordinates'].values)  # (N_test, d)

    # --- 2) Fit scalers on combined supervised + unpaired sets ---
    scaler_X = StandardScaler().fit(np.vstack((X_sup, X_img)))
    scaler_Y = StandardScaler().fit(np.vstack((Y_sup, Y_gene)))

    # --- 3) Transform all sets into standardized space ---
    Xs_sup  = scaler_X.transform(X_sup)
    Xs_img  = scaler_X.transform(X_img)
    Xs_test = scaler_X.transform(X_test)
    Ys_sup  = scaler_Y.transform(Y_sup)
    Ys_gene = scaler_Y.transform(Y_gene)

    Xs_sup  = np.asarray(Xs_sup,  dtype=np.float64)
    Xs_img  = np.asarray(Xs_img,  dtype=np.float64)
    Xs_test = np.asarray(Xs_test, dtype=np.float64)
    Ys_sup  = np.asarray(Ys_sup,  dtype=np.float64)
    Ys_gene = np.asarray(Ys_gene, dtype=np.float64)

    # --- 4) Choose sigma if not provided ---
    if sigma is None:
        subset = Xs_sup[np.random.choice(len(Xs_sup), min(len(Xs_sup), 200), replace=False)]
        d2 = np.sum((subset[:, None, :] - subset[None, :, :])**2, axis=2).ravel()
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med)
    gamma = 1.0 / (2 * sigma**2)

    # --- 5) Prepare dummy y for KMM and compute weights ---
    N_sup = Xs_sup.shape[0]
    dummy_y = np.zeros(N_sup)

    kmm_x = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False)
    # supply named args so y is dummy_y and Xt is the image-only target
    kmm_x.fit(X=Xs_sup, y=dummy_y, Xt=Xs_img)
    w_x = kmm_x.weights_

    kmm_y = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False)
    kmm_y.fit(X=Ys_sup, y=dummy_y, Xt=Ys_gene)
    w_y = kmm_y.weights_

    # --- 6) Combine and renormalize weights ---
    w = w_x * w_y
    w *= (len(w) / np.sum(w))

    # --- 7) Fit weighted Kernel Ridge Regression per output dim ---
    preds = []
    for dim in range(Ys_sup.shape[1]):
        kr = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        kr.fit(Xs_sup, Ys_sup[:, dim], sample_weight=w)
        preds.append(kr.predict(Xs_test))
    Y_pred = np.vstack(preds).T

    # --- 8) Inverse-transform and return ---
    Y_pred = scaler_Y.inverse_transform(Y_pred)
    Y_true = np.vstack(inference_df['gene_coordinates'].values)
    return Y_pred, Y_true


def reversed_kernel_mean_matching_regression(
    gene_df,
    image_df,
    supervised_df,
    inference_df,
    alpha: float = 1e-2,
    kmm_B: int = 1000,
    kmm_eps: float = 1e-3,
    sigma: float | None = None,
    random_state: int = 0
):
    # 1) Extract raw arrays
    X_gene_unp = np.vstack(gene_df['gene_coordinates'].values)
    Y_img_unp  = np.vstack(image_df['morph_coordinates'].values)
    X_sup      = np.vstack(supervised_df['gene_coordinates'].values)
    Y_sup      = np.vstack(supervised_df['morph_coordinates'].values)
    X_test     = np.vstack(inference_df['gene_coordinates'].values)

    # 2) Standardize over supervised + unpaired
    scaler_X = StandardScaler().fit(np.vstack((X_sup, X_gene_unp)))
    scaler_Y = StandardScaler().fit(np.vstack((Y_sup, Y_img_unp)))
    Xs_sup      = scaler_X.transform(X_sup)
    Xs_gene_unp = scaler_X.transform(X_gene_unp)
    Ys_sup      = scaler_Y.transform(Y_sup)
    Ys_img_unp  = scaler_Y.transform(Y_img_unp)
    Xs_test     = scaler_X.transform(X_test)

    Xs_sup      = np.asarray(Xs_sup,      dtype=np.float64)
    Xs_gene_unp = np.asarray(Xs_gene_unp, dtype=np.float64)
    Ys_sup      = np.asarray(Ys_sup,      dtype=np.float64)
    Ys_img_unp  = np.asarray(Ys_img_unp,  dtype=np.float64)
    Xs_test     = np.asarray(Xs_test,     dtype=np.float64)

    # 3) Choose sigma if not provided
    if sigma is None:
        subset = Xs_sup[np.random.choice(len(Xs_sup), min(len(Xs_sup), 200), replace=False)]
        d2 = np.sum((subset[:, None, :] - subset[None, :, :])**2, axis=2).ravel()
        med = np.median(d2[d2 > 0])
        sigma = np.sqrt(med)
    gamma = 1.0 / (2 * sigma**2)

    # 4) Prepare dummy y for KMM
    N_sup = Xs_sup.shape[0]
    dummy_y = np.zeros(N_sup)

    # 5) KMM to match supervised → unpaired distributions
    kmm_x = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False, random_state=random_state)
    kmm_x.fit(X=Xs_sup, y=dummy_y, Xt=Xs_gene_unp)
    w_x = kmm_x.weights_

    kmm_y = KMM(kernel="rbf", B=kmm_B, eps=kmm_eps, gamma=gamma, verbose=False, random_state=random_state)
    kmm_y.fit(X=Ys_sup, y=dummy_y, Xt=Ys_img_unp)
    w_y = kmm_y.weights_

    # 6) Combine and normalize weights
    w = w_x * w_y
    w *= (len(w) / np.sum(w))

    # 7) Fit weighted Kernel Ridge Regression per output dimension
    preds = []
    for dim in range(Ys_sup.shape[1]):
        kr = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        kr.fit(Xs_sup, Ys_sup[:, dim], sample_weight=w)
        preds.append(kr.predict(Xs_test))
    Y_pred_s = np.vstack(preds).T

    # 8) Inverse-transform and return
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)
    Y_true = np.vstack(inference_df['morph_coordinates'].values)
    return Y_pred, Y_true


import numpy as np
from sklearn.cluster import KMeans

def gaussian_logpdf(X, mu, var, eps=1e-8):
    var = var + eps
    d = X.shape[1]
    diff = X - mu
    exponent = -0.5 * np.sum(diff**2, axis=1) / var
    log_norm = 0.5 * d * np.log(2 * np.pi * var)
    return exponent - log_norm

def em_regression(
    supervised_df,
    image_df,
    gene_df,
    inference_df,
    n_components=3,
    max_iter=2000,
    tol=1e-4,
    eps=1e-8
):
    # 1) Stack
    X_sup = np.vstack(supervised_df['morph_coordinates'])
    Y_sup = np.vstack(supervised_df['gene_coordinates'])
    X_xo  = np.vstack(image_df['morph_coordinates']) if len(image_df) else np.empty((0, X_sup.shape[1]))
    Y_yo  = np.vstack(gene_df['gene_coordinates'])    if len(gene_df)  else np.empty((0, Y_sup.shape[1]))

    n_sup, d_x = X_sup.shape
    n_xo = X_xo.shape[0]
    n_yo = Y_yo.shape[0]
    d_y = Y_sup.shape[1]
    K   = n_components

    # 2) Init with KMeans on the supervised subset
    mu_x = KMeans(K, random_state=0).fit(X_sup).cluster_centers_
    mu_y = KMeans(K, random_state=0).fit(Y_sup).cluster_centers_
    var_x = np.full(K, np.mean(np.var(X_sup, axis=0)) + eps)
    var_y = np.full(K, np.mean(np.var(Y_sup, axis=0)) + eps)
    pi    = np.full(K, 1.0/K)

    # 3) EM
    for _ in range(max_iter):
        # --- E-step: build log-responsibilities ---
        log_r_sup = np.stack([
            np.log(pi[k] + eps)
            + gaussian_logpdf(X_sup, mu_x[k], var_x[k], eps)
            + gaussian_logpdf(Y_sup, mu_y[k], var_y[k], eps)
            for k in range(K)
        ], axis=1)
        log_r_sup -= log_r_sup.max(axis=1, keepdims=True)
        r_sup = np.exp(log_r_sup)
        r_sup /= (r_sup.sum(axis=1, keepdims=True) + eps)

        if n_xo>0:
            log_r_xo = np.stack([
                np.log(pi[k] + eps)
                + gaussian_logpdf(X_xo, mu_x[k], var_x[k], eps)
                for k in range(K)
            ], axis=1)
            log_r_xo -= log_r_xo.max(axis=1, keepdims=True)
            r_xo = np.exp(log_r_xo)
            r_xo /= (r_xo.sum(axis=1, keepdims=True) + eps)
        else:
            r_xo = np.zeros((0,K))

        if n_yo>0:
            log_r_yo = np.stack([
                np.log(pi[k] + eps)
                + gaussian_logpdf(Y_yo, mu_y[k], var_y[k], eps)
                for k in range(K)
            ], axis=1)
            log_r_yo -= log_r_yo.max(axis=1, keepdims=True)
            r_yo = np.exp(log_r_yo)
            r_yo /= (r_yo.sum(axis=1, keepdims=True) + eps)
        else:
            r_yo = np.zeros((0,K))

        # --- M-step ---
        Nk = r_sup.sum(axis=0) + r_xo.sum(axis=0) + r_yo.sum(axis=0)
        pi_new = Nk / (n_sup + n_xo + n_yo)

        mu_x_new = np.zeros_like(mu_x)
        mu_y_new = np.zeros_like(mu_y)
        var_x_new = np.zeros_like(var_x)
        var_y_new = np.zeros_like(var_y)

        for k in range(K):
            w_x = r_sup[:,k].sum() + r_xo[:,k].sum()
            if w_x>0:
                mu_x_new[k] = (
                    (r_sup[:,k,None]*X_sup).sum(0)
                  + (r_xo[:,k,None]*X_xo).sum(0)
                )/(w_x+eps)

            w_y = r_sup[:,k].sum() + r_yo[:,k].sum()
            if w_y>0:
                mu_y_new[k] = (
                    (r_sup[:,k,None]*Y_sup).sum(0)
                  + (r_yo[:,k,None]*Y_yo).sum(0)
                )/(w_y+eps)

            if w_x>0:
                dx_sup = X_sup - mu_x_new[k]
                dx_xo  = X_xo  - mu_x_new[k] if n_xo>0 else np.zeros((0,d_x))
                sx = (
                    (r_sup[:,k]*np.sum(dx_sup**2,axis=1)).sum()
                  + (r_xo[:,k]*np.sum(dx_xo**2,axis=1)).sum()
                )
                var_x_new[k] = sx/(d_x*(w_x+eps))+eps

            if w_y>0:
                dy_sup = Y_sup - mu_y_new[k]
                dy_yo  = Y_yo  - mu_y_new[k] if n_yo>0 else np.zeros((0,d_y))
                sy = (
                    (r_sup[:,k]*np.sum(dy_sup**2,axis=1)).sum()
                  + (r_yo[:,k]*np.sum(dy_yo**2,axis=1)).sum()
                )
                var_y_new[k] = sy/(d_y*(w_y+eps))+eps

        if (np.max(np.abs(pi_new-pi))<tol
        and np.max(np.abs(mu_x_new-mu_x))<tol
        and np.max(np.abs(mu_y_new-mu_y))<tol):
            pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new
            break

        pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new

    # 4) Inference
    X_test = np.vstack(inference_df['morph_coordinates'])
    Y_true = np.vstack(inference_df['gene_coordinates'])

    log_resp = np.stack([
        np.log(pi[k] + eps)
        + gaussian_logpdf(X_test, mu_x[k], var_x[k], eps)
        for k in range(K)
    ], axis=1)
    log_resp -= log_resp.max(axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp /= (resp.sum(axis=1, keepdims=True) + eps)

    Y_pred = resp.dot(mu_y)
    return Y_pred, Y_true


def reversed_em_regression(
    gene_df,
    image_df,
    supervised_df,
    inference_df,
    n_components=3,
    max_iter=2000,
    tol=1e-4,
    eps=1e-8
):
    # exactly the same but swap roles of X<->Y
    # X_sup = gene → Y_sup = image, etc.
    X_sup = np.vstack(supervised_df['gene_coordinates'])
    Y_sup = np.vstack(supervised_df['morph_coordinates'])
    X_xo  = np.vstack(gene_df['gene_coordinates'])    if len(gene_df)  else np.empty((0,X_sup.shape[1]))
    Y_yo  = np.vstack(image_df['morph_coordinates'])  if len(image_df) else np.empty((0,Y_sup.shape[1]))

    n_sup, d_x = X_sup.shape
    n_xo = X_xo.shape[0]
    n_yo = Y_yo.shape[0]
    d_y = Y_sup.shape[1]
    K   = n_components

    mu_x = KMeans(K, random_state=0).fit(X_sup).cluster_centers_
    mu_y = KMeans(K, random_state=0).fit(Y_sup).cluster_centers_
    var_x = np.full(K, np.mean(np.var(X_sup,axis=0))+eps)
    var_y = np.full(K, np.mean(np.var(Y_sup,axis=0))+eps)
    pi    = np.full(K, 1.0/K)

    # same EM loop as above...
    for _ in range(max_iter):
        log_r_sup = np.stack([
            np.log(pi[k]+eps)
            + gaussian_logpdf(X_sup, mu_x[k], var_x[k], eps)
            + gaussian_logpdf(Y_sup, mu_y[k], var_y[k], eps)
            for k in range(K)
        ],axis=1)
        log_r_sup -= log_r_sup.max(axis=1,keepdims=True)
        r_sup = np.exp(log_r_sup)
        r_sup /= (r_sup.sum(axis=1,keepdims=True)+eps)

        if n_xo>0:
            log_r_xo = np.stack([
                np.log(pi[k]+eps)
                + gaussian_logpdf(X_xo, mu_x[k], var_x[k], eps)
                for k in range(K)
            ],axis=1)
            log_r_xo -= log_r_xo.max(axis=1,keepdims=True)
            r_xo = np.exp(log_r_xo)
            r_xo /= (r_xo.sum(axis=1,keepdims=True)+eps)
        else:
            r_xo = np.zeros((0,K))

        if n_yo>0:
            log_r_yo = np.stack([
                np.log(pi[k]+eps)
                + gaussian_logpdf(Y_yo, mu_y[k], var_y[k], eps)
                for k in range(K)
            ],axis=1)
            log_r_yo -= log_r_yo.max(axis=1,keepdims=True)
            r_yo = np.exp(log_r_yo)
            r_yo /= (r_yo.sum(axis=1,keepdims=True)+eps)
        else:
            r_yo = np.zeros((0,K))

        Nk = r_sup.sum(axis=0) + r_xo.sum(axis=0) + r_yo.sum(axis=0)
        pi_new = Nk/(n_sup+n_xo+n_yo)

        mu_x_new = np.zeros_like(mu_x)
        mu_y_new = np.zeros_like(mu_y)
        var_x_new = np.zeros_like(var_x)
        var_y_new = np.zeros_like(var_y)

        for k in range(K):
            w_x = r_sup[:,k].sum() + r_xo[:,k].sum()
            if w_x>0:
                mu_x_new[k] = (
                    (r_sup[:,k,None]*X_sup).sum(0)
                  + (r_xo[:,k,None]*X_xo).sum(0)
                )/(w_x+eps)

            w_y = r_sup[:,k].sum() + r_yo[:,k].sum()
            if w_y>0:
                mu_y_new[k] = (
                    (r_sup[:,k,None]*Y_sup).sum(0)
                  + (r_yo[:,k,None]*Y_yo).sum(0)
                )/(w_y+eps)

            if w_x>0:
                dx_sup = X_sup - mu_x_new[k]
                dx_xo  = X_xo  - mu_x_new[k] if n_xo>0 else np.zeros((0,d_x))
                sx = (
                    (r_sup[:,k]*np.sum(dx_sup**2,axis=1)).sum()
                  + (r_xo[:,k]*np.sum(dx_xo**2,axis=1)).sum()
                )
                var_x_new[k] = sx/(d_x*(w_x+eps))+eps

            if w_y>0:
                dy_sup = Y_sup - mu_y_new[k]
                dy_yo  = Y_yo  - mu_y_new[k] if n_yo>0 else np.zeros((0,d_y))
                sy = (
                    (r_sup[:,k]*np.sum(dy_sup**2,axis=1)).sum()
                  + (r_yo[:,k]*np.sum(dy_yo**2,axis=1)).sum()
                )
                var_y_new[k] = sy/(d_y*(w_y+eps))+eps

        if (np.max(np.abs(pi_new-pi))<tol
        and np.max(np.abs(mu_x_new-mu_x))<tol
        and np.max(np.abs(mu_y_new-mu_y))<tol):
            pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new
            break

        pi, mu_x, mu_y, var_x, var_y = pi_new, mu_x_new, mu_y_new, var_x_new, var_y_new

    X_test = np.vstack(inference_df['gene_coordinates'])
    Y_true = np.vstack(inference_df['morph_coordinates'])

    log_resp = np.stack([
        np.log(pi[k]+eps)
        + gaussian_logpdf(X_test, mu_x[k], var_x[k], eps)
        for k in range(K)
    ],axis=1)
    log_resp -= log_resp.max(axis=1,keepdims=True)
    resp = np.exp(log_resp)
    resp /= (resp.sum(axis=1,keepdims=True)+eps)

    Y_pred = resp.dot(mu_y)
    return Y_pred, Y_true




import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

# Optional: POT, else fall back to local Sinkhorn
try:
    import ot
    _HAS_POT = True
except Exception:
    _HAS_POT = False

# ---------- small helpers ----------

def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # (n,d) vs (m,d) -> (n,m) squared euclidean
    A2 = (A * A).sum(axis=1, keepdims=True)   # (n,1)
    B2 = (B * B).sum(axis=1, keepdims=True).T # (1,m)
    return A2 + B2 - 2.0 * (A @ B.T)

def _choose_epsilon(C: np.ndarray, frac: float = 0.05, rng: int | None = None) -> float:
    # pick ε as a small fraction of a robust cost scale; train-only!
    rs = np.random.default_rng(rng)
    n, m = C.shape
    # sample up to 20 entries for robustness
    idx_n = rs.integers(0, n, size=min(n, 20))
    idx_m = rs.integers(0, m, size=min(m, 20))
    sample = C[np.ix_(idx_n, idx_m)].ravel()
    med = np.median(sample) if sample.size else 1.0
    return float(max(1e-4, frac * med))

def _sinkhorn_balanced(a, b, C, eps, max_iter=2000, tol=1e-6):
    # lightweight log-domain Sinkhorn (balanced)
    K = np.exp(-C / eps, dtype=np.float64)  # (n,m)
    u = np.ones_like(a, dtype=np.float64)
    v = np.ones_like(b, dtype=np.float64)
    # avoid divide by 0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    for _ in range(max_iter):
        u_prev = u
        Ku = K @ v + 1e-300
        u = a / Ku
        Kt_u = K.T @ u + 1e-300
        v = b / Kt_u
        if np.linalg.norm(u - u_prev, 1) < tol * (len(u) + 1e-12):
            break
    P = (u[:, None] * K) * v[None, :]
    return P

def _maybe_pca_train_only(X_train: np.ndarray, X_eval: np.ndarray, n_components: int | None):
    if n_components is None:
        return None, X_train, X_eval
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    Xtr = pca.fit_transform(X_train)
    Xev = pca.transform(X_eval)
    return pca, Xtr, Xev

def _subsample_pool(Y_pool: np.ndarray, max_size: int | None, rng: int | None):
    if (max_size is None) or (Y_pool.shape[0] <= max_size):
        idx = np.arange(len(Y_pool))
        return Y_pool, idx
    rs = np.random.default_rng(rng)
    idx = rs.choice(Y_pool.shape[0], size=max_size, replace=False)
    return Y_pool[idx], idx


def eot_barycentric_regression(
    image_df,
    gene_df,
    supervised_df,
    inference_df,
    *,
    ridge_alpha: float = 1e-2,
    eps: float | None = None,
    max_iter: int = 2000,  # usually high enough for convergence guarantee
    tol: float = 1e-9,    # strict stopping criterion
    use_pot: bool = True,
    random_state: int = 1,
    pca_x: int | None = None,
    pca_y: int | None = None,
    max_transport_y: int | None = 2000  # cap for compute parity
):
    # ---------- extract arrays ----------
    X_pool  = np.vstack(image_df['morph_coordinates'].values)      if len(image_df)      else np.zeros((0, supervised_df['morph_coordinates'].iloc[0].__len__()))
    Y_pool  = np.vstack(gene_df['gene_coordinates'].values)        if len(gene_df)       else np.zeros((0, supervised_df['gene_coordinates'].iloc[0].__len__()))
    X_sup   = np.vstack(supervised_df['morph_coordinates'].values) if len(supervised_df) else np.zeros((0, X_pool.shape[1] if X_pool.size else 0))
    Y_sup   = np.vstack(supervised_df['gene_coordinates'].values)  if len(supervised_df) else np.zeros((0, Y_pool.shape[1] if Y_pool.size else 0))
    X_test  = np.vstack(inference_df['morph_coordinates'].values)
    Y_true  = np.vstack(inference_df['gene_coordinates'].values)

    if Y_pool.shape[0] == 0:
        raise ValueError("gene_df is empty; EOT needs a Y-pool to transport onto.")

    # ---------- train-only scaling ----------
    X_train_for_scaler = np.vstack([X_pool, X_sup]) if (X_pool.size or X_sup.size) else X_test
    Y_train_for_scaler = np.vstack([Y_pool, Y_sup]) if (Y_pool.size or Y_sup.size) else Y_true

    scal_X = StandardScaler().fit(X_train_for_scaler)
    scal_Y = StandardScaler().fit(Y_train_for_scaler)

    Xs_sup  = scal_X.transform(X_sup)
    Ys_sup  = scal_Y.transform(Y_sup)
    Xs_pool = scal_X.transform(X_pool)
    Ys_pool = scal_Y.transform(Y_pool)
    Xs_test = scal_X.transform(X_test)

    # ---------- optional train-only PCA ----------
    pX, _, Xs_test_p = _maybe_pca_train_only(
        np.vstack([Xs_sup, Xs_pool]) if (Xs_sup.size or Xs_pool.size) else Xs_test,
        Xs_test,
        pca_x
    )
    if pX is not None:
        Xs_sup  = pX.transform(Xs_sup)  if Xs_sup.size  else Xs_sup
        Xs_pool = pX.transform(Xs_pool) if Xs_pool.size else Xs_pool

    pY, _, Ys_pool_p = _maybe_pca_train_only(
        np.vstack([Ys_sup, Ys_pool]) if (Ys_sup.size or Ys_pool.size) else Y_true,
        Ys_pool,
        pca_y
    )
    if pY is not None:
        Ys_sup = pY.transform(Ys_sup) if Ys_sup.size else Ys_sup

    # ---------- ridge map W (X->Y) ----------
    if Xs_sup.shape[0] == 0:
        d_in  = Xs_test_p.shape[1] if pX is not None else Xs_test.shape[1]
        d_out = Ys_pool_p.shape[1] if pY is not None else Ys_pool.shape[1]
        W = np.zeros((d_in, d_out), dtype=np.float64)
    else:
        rr = Ridge(alpha=ridge_alpha, fit_intercept=False, random_state=random_state)
        rr.fit(Xs_sup, Ys_sup)
        W = rr.coef_.T.astype(np.float64, copy=False)

    # ---------- subsample Y-pool for parity ----------
    Ys_pool_use, _ = _subsample_pool(Ys_pool_p if pY is not None else Ys_pool,
                                     max_transport_y, random_state)

    # ---------- OT cost ----------
    X_to_Y = (Xs_test_p if pX is not None else Xs_test) @ W
    C = _pairwise_sq_dists(X_to_Y.astype(np.float64, copy=False),
                           Ys_pool_use.astype(np.float64, copy=False))
    n, m = C.shape
    a = np.full(n, 1.0/n, dtype=np.float64)
    b = np.full(m, 1.0/m, dtype=np.float64)

    eps = _choose_epsilon(C, frac=0.05, rng=random_state)

    # ---------- Sinkhorn ----------
    if use_pot and _HAS_POT:
        P = ot.sinkhorn(a, b, C, reg=eps, numItermax=max_iter,
                stopThr=tol)
    else:
        P = _sinkhorn_balanced(a, b, C, eps, max_iter=max_iter, tol=tol)

    # ---------- barycentric mapping ----------
    rows = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    Y_pred_s = (P @ Ys_pool_use) / rows

    # inverse PCA & scaling
    if pY is not None:
        Y_pred_std = pY.inverse_transform(Y_pred_s)
    else:
        Y_pred_std = Y_pred_s
    Y_pred = scal_Y.inverse_transform(Y_pred_std)
    return Y_pred, Y_true


def reversed_eot_barycentric_regression(
    gene_df,
    image_df,
    supervised_df,
    inference_df,
    *,
    ridge_alpha: float = 1e-2,
    eps: float = 10,
    max_iter: int = 2000,
    tol: float = 1e-6,
    use_pot: bool = True,
    random_state: int = 0,
    pca_x: int | None = None,
    pca_y: int | None = None,
    max_transport_y: int | None = 2000,
):
    """
    Entropic OT + barycentric mapping (genes -> images), with train-only preprocessing and capped OT.
    Train-only:
      Z_train = supervised.gene ∪ gene_df.gene
      X_train = supervised.morph ∪ image_df.morph
    Test:
      Z_test  = inference.gene  (labels in inference.morph)
    """
    Z_pool = np.vstack(gene_df['gene_coordinates'].values)              if len(gene_df)       else np.zeros((0, supervised_df['gene_coordinates'].iloc[0].__len__()))
    X_pool = np.vstack(image_df['morph_coordinates'].values)            if len(image_df)      else np.zeros((0, supervised_df['morph_coordinates'].iloc[0].__len__()))
    Z_sup  = np.vstack(supervised_df['gene_coordinates'].values)        if len(supervised_df) else np.zeros((0, Z_pool.shape[1] if Z_pool.size else 0))
    X_sup  = np.vstack(supervised_df['morph_coordinates'].values)       if len(supervised_df) else np.zeros((0, X_pool.shape[1] if X_pool.size else 0))

    Z_test = np.vstack(inference_df['gene_coordinates'].values)
    X_true = np.vstack(inference_df['morph_coordinates'].values)

    if X_pool.shape[0] == 0:
        raise ValueError("image_df is empty; reversed EOT needs an image pool to transport onto.")

    # ---------- train-only scaling ----------
    Z_train_for_scaler = np.vstack([Z_pool, Z_sup]) if Z_pool.size or Z_sup.size else Z_test
    X_train_for_scaler = np.vstack([X_pool, X_sup]) if X_pool.size or X_sup.size else X_true

    scal_Z = StandardScaler().fit(Z_train_for_scaler)
    scal_X = StandardScaler().fit(X_train_for_scaler)

    Zs_sup  = scal_Z.transform(Z_sup)
    Xs_sup  = scal_X.transform(X_sup)
    Zs_pool = scal_Z.transform(Z_pool)
    Xs_pool = scal_X.transform(X_pool)
    Zs_test = scal_Z.transform(Z_test)

    # ---------- optional train-only PCA ----------
    pZ, Zs_train_dummy, Zs_test_p = _maybe_pca_train_only(
        np.vstack([Zs_sup, Zs_pool]) if (Zs_sup.size or Zs_pool.size) else Zs_test,
        Zs_test,
        pca_x
    )
    if pZ is not None:
        Zs_sup  = pZ.transform(Zs_sup)  if Zs_sup.size  else Zs_sup
        Zs_pool = pZ.transform(Zs_pool) if Zs_pool.size else Zs_pool

    pX, Xs_train_dummy, Xs_pool_p = _maybe_pca_train_only(
        np.vstack([Xs_sup, Xs_pool]) if (Xs_sup.size or Xs_pool.size) else X_true,
        Xs_pool,
        pca_y
    )
    if pX is not None:
        Xs_sup = pX.transform(Xs_sup) if Xs_sup.size else Xs_sup

    # ---------- ridge map W (genes -> images) ----------
    if Zs_sup.shape[0] == 0:
        d_in  = Zs_test_p.shape[1] if pZ is not None else Zs_test.shape[1]
        d_out = Xs_pool_p.shape[1] if pX is not None else Xs_pool.shape[1]
        W = np.zeros((d_in, d_out), dtype=np.float64)
    else:
        rr = Ridge(alpha=ridge_alpha, fit_intercept=False, random_state=random_state)
        rr.fit(Zs_sup, Xs_sup)
        W = rr.coef_.T.astype(np.float64, copy=False)

    # ---------- Y-pool cap (now Y = images) ----------
    Xs_pool_use, idx_pool = _subsample_pool(Xs_pool_p if pX is not None else Xs_pool,
                                            max_transport_y, random_state)

    # ---------- OT cost on image space ----------
    Z_to_X = (Zs_test_p if pZ is not None else Zs_test) @ W
    C = _pairwise_sq_dists(Z_to_X.astype(np.float64, copy=False),
                           Xs_pool_use.astype(np.float64, copy=False))
    n, m = C.shape
    a = np.full(n, 1.0/n, dtype=np.float64)
    b = np.full(m, 1.0/m, dtype=np.float64)

    eps = _choose_epsilon(C, frac=0.05, rng=random_state)

    if use_pot and _HAS_POT:
        P = ot.sinkhorn(a, b, C, reg=eps, numItermax=max_iter, stopThr=tol,method="sinkhorn_log")
    else:
        P = _sinkhorn_balanced(a, b, C, eps, max_iter=max_iter, tol=tol)

    # ---------- barycentric X (standardized/PCA’d) ----------
    rows = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    X_pred_s = (P @ Xs_pool_use) / rows

    # inverse PCA/scale
    if pX is not None:
        X_pred_std = pX.inverse_transform(X_pred_s)
    else:
        X_pred_std = X_pred_s
    X_pred = scal_X.inverse_transform(X_pred_std)
    return X_pred, X_true


def gw_metric_alignment_regression(
    image_df,
    gene_df,
    supervised_df,
    inference_df,
    epsilon=None,
    max_iter=2000,
    tol=1e-9,
    random_state=0
):
    # --- 1) Stack & standardize ---
    X_img  = np.vstack(image_df['morph_coordinates'].values)      if len(image_df)     else None
    Y_gene = np.vstack(gene_df['gene_coordinates'].values)        if len(gene_df)      else None
    X_sup  = np.vstack(supervised_df['morph_coordinates'].values) if len(supervised_df) else None
    Y_sup  = np.vstack(supervised_df['gene_coordinates'].values)  if len(supervised_df) else None
    X_test = np.vstack(inference_df['morph_coordinates'].values)
    Y_true = np.vstack(inference_df['gene_coordinates'].values)

    if Y_gene is None or Y_gene.shape[0] == 0:
        raise ValueError("gene_df is empty; GW needs the output-only pool to align to.")

    X_for_scaler = np.vstack([arr for arr in (X_sup, X_img, X_test) if arr is not None and arr.size > 0])
    Y_for_scaler = np.vstack([arr for arr in (Y_sup, Y_gene) if arr is not None and arr.size > 0])

    scaler_X = StandardScaler().fit(X_for_scaler)
    scaler_Y = StandardScaler().fit(Y_for_scaler)

    Xs_test = scaler_X.transform(X_test).astype(np.float64, copy=False)
    Ys_gene = scaler_Y.transform(Y_gene).astype(np.float64, copy=False)

    # --- 2) Intra-domain squared distances ---
    Cx = _pairwise_sq_dists(Xs_test, Xs_test)
    Cy = _pairwise_sq_dists(Ys_gene, Ys_gene)
    np.fill_diagonal(Cx, 0.0)
    np.fill_diagonal(Cy, 0.0)

    # Rescale so medians ≈ 1.0 for numeric stability
    scale_x = np.median(Cx[Cx > 0])
    scale_y = np.median(Cy[Cy > 0])
    if scale_x > 0: Cx /= scale_x
    if scale_y > 0: Cy /= scale_y

    # --- 3) Solve entropic GW ---
    n, m = Xs_test.shape[0], Ys_gene.shape[0]
    a = np.full(n, 1.0/n, dtype=np.float64)
    b = np.full(m, 1.0/m, dtype=np.float64)

    if epsilon is None:
        med = 0.5 * (np.median(Cx) + np.median(Cy))
        epsilon_ = max(1e-6, 0.1 * float(med))  # bumped from 0.05 to 0.1
    else:
        epsilon_ = float(epsilon)

    P, log = ot.gromov.entropic_gromov_wasserstein(
        Cx, Cy, a, b,
        loss_fun="square_loss",
        epsilon=epsilon_,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
        log=True
    )

    # --- 4) Barycentric prediction ---
    row_sums = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    Y_pred_s = (P @ Ys_gene) / row_sums
    Y_pred = scaler_Y.inverse_transform(Y_pred_s)
    return Y_pred, Y_true


def reversed_gw_metric_alignment_regression(
    gene_df,
    image_df,
    supervised_df,
    inference_df,
    epsilon=None,
    max_iter=2000,
    tol=1e-9,
    random_state=0
):

    Z_sup = np.vstack(supervised_df['gene_coordinates'].values)   if len(supervised_df) else None
    X_sup = np.vstack(supervised_df['morph_coordinates'].values)  if len(supervised_df) else None
    Z_pool = np.vstack(gene_df['gene_coordinates'].values)        if len(gene_df)       else None
    X_pool = np.vstack(image_df['morph_coordinates'].values)      if len(image_df)      else None
    Z_test = np.vstack(inference_df['gene_coordinates'].values)
    X_true = np.vstack(inference_df['morph_coordinates'].values)

    if X_pool is None or X_pool.shape[0] == 0:
        raise ValueError("image_df is empty; reversed GW needs the morph (input-only) pool to align to.")

    scaler_Z = StandardScaler().fit(np.vstack([arr for arr in (Z_sup, Z_pool, Z_test) if arr is not None and arr.size > 0]))
    scaler_X = StandardScaler().fit(np.vstack([arr for arr in (X_sup, X_pool) if arr is not None and arr.size > 0]))

    Zs_test = scaler_Z.transform(Z_test).astype(np.float64, copy=False)
    Xs_pool = scaler_X.transform(X_pool).astype(np.float64, copy=False)

    # Intra-domain squared distances
    Cg = _pairwise_sq_dists(Zs_test, Zs_test)
    Cm = _pairwise_sq_dists(Xs_pool, Xs_pool)
    np.fill_diagonal(Cg, 0.0)
    np.fill_diagonal(Cm, 0.0)

    # Rescale
    scale_g = np.median(Cg[Cg > 0])
    scale_m = np.median(Cm[Cm > 0])
    if scale_g > 0: Cg /= scale_g
    if scale_m > 0: Cm /= scale_m

    n, m = Zs_test.shape[0], Xs_pool.shape[0]
    a = np.full(n, 1.0/n, dtype=np.float64)
    b = np.full(m, 1.0/m, dtype=np.float64)
    if epsilon is None:
        med = 0.5 * (np.median(Cg) + np.median(Cm))
        eps = max(1e-6, 0.1 * float(med))
    else:
        eps = float(epsilon)

    P = ot.gromov.entropic_gromov_wasserstein(
        Cg, Cm, a, b,
        loss_fun="square_loss",
        epsilon=eps,
        max_iter=max_iter,
        tol=tol,
        verbose=False
    )

    rows = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    X_pred_s = (P @ Xs_pool) / rows
    X_pred = scaler_X.inverse_transform(X_pred_s)
    return X_pred, X_true
