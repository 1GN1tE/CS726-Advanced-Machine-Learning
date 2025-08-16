import torch
import numpy as np  
from sklearn.datasets import (
    make_circles,
    make_moons,
)
import utils

def normalize_data(X):
    mmin = X.min(dim=0)[0]
    mmax = X.max(dim=0)[0]
    return 2*(X - mmin) / (mmax - mmin) - 1

def generate_many_circle_data(N_SAMPLES, N_GROUPS, scale):
    # group assignment
    group_id = np.random.randint(low=0, high=N_GROUPS, size=N_SAMPLES)
    linspace_theta = np.linspace(0, 2*np.pi, N_GROUPS + 1)[:-1]; linspace_theta * 180 / np.pi
    group_centers = np.vstack([np.cos(linspace_theta), np.sin(linspace_theta)]).T
    means = group_centers[group_id]
    data = np.random.randn(N_SAMPLES,2)*scale + means
    return data, group_id

def load_dataset(dataset_name):
    if dataset_name == 'albatross':
        X = torch.from_numpy(np.load('data/albatross.npy')).to(torch.float32)
        B, W, L =  X.shape
        X = X.view(B, -1)
        # y = None
        y = torch.ones_like(X)
        X = normalize_data(X)
        return X, y
    
    elif dataset_name == 'moons':
        X, y = make_moons(n_samples=8000, noise=0.04, random_state=42)

    elif dataset_name == 'circles':
        X, y = make_circles(n_samples=8000, noise=0.04, random_state=42, factor=0.7)

    elif dataset_name == 'manycircles':
        X, y = generate_many_circle_data(8000, 8, 0.250)

    elif dataset_name == 'helix':
        X = np.load('data/helix.npy')
        utils.seed_everything(42)
        y = np.random.binomial(1, (np.sin(4 * X[:, 0] * X[:, 1] * X[:, 2])+1)/2)

    elif dataset_name == 'blobs':
        utils.seed_everything(42)
        N_SAMPLES = 8000
        PRIOR_1 = 0.2
        N_POS = np.random.binomial(N_SAMPLES, p=PRIOR_1)
        N_NEG = N_SAMPLES - N_POS

        PRIOR_1_M = 0.2  # prior for mix
        N_POS_M1 = np.random.binomial(N_POS, p=PRIOR_1_M)
        N_POS_M2 = N_POS - N_POS_M1

        f = np.array([[0.005, 0], [0, 0.005]])
        g = np.array([[0.00, 0.004], [0.004, 0.00]])
        X_pos_M1 = np.random.multivariate_normal(
            mean=[0.2, 0.3], cov=2 * f + g, size=N_POS_M1
        )
        X_pos_M2 = np.random.multivariate_normal(
            mean=[0.8, 0.8], cov=f - g, size=N_POS_M2
        )
        X_pos = np.vstack((X_pos_M1, X_pos_M2))
        y_pos = np.ones(N_POS)

        X_neg = np.random.multivariate_normal(mean=[0.3, 0.4], cov=f, size=N_NEG)
        y_neg = np.zeros(N_NEG)

        X = np.vstack((X_pos, X_neg))
        y = np.hstack((y_pos, y_neg))
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.long)
    X = normalize_data(X)
    return X, y
    
