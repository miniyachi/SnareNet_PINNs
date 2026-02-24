import numpy as np
import torch
from .constraints import PDEConstraint

"""
Helper function for creating a PDEConstraint object for SnareNet.

INPUT:
- pde_name: string; name of the PDE problem
- pde_coefs: dictionary containing coefficients of the PDE
- x_range: list of size 2; lower and upper bounds of spatial variable x
- t_range: list of size 2; lower and upper bounds of temporal variable t
OUTPUT:
- pde_constraint: PDEConstraint instance
"""
def get_pde_constraint(pde_name, pde_coefs, x_range, t_range):
    return PDEConstraint(pde_name, pde_coefs, x_range, t_range)

"""
Helper function for transforming standard PINN data to paired format for SnareNet.

This function creates paired inputs (x1, x2, t) from standard PINN data:
- For residual points: x1 = x2 (same point, enforces PDE)
- For boundary points: x1 and x2 are boundary values (enforces BC)
- For initial points: x1 = x2 at t=0 (enforces IC)

INPUT:
- x: tuple of (x_res, x_left, x_upper, x_lower) from get_data
- t: tuple of (t_res, t_left, t_upper, t_lower) from get_data
OUTPUT:
- x1_all: tensor; first spatial coordinate
- x2_all: tensor; second spatial coordinate
- t_all: tensor; temporal coordinate
- indices: dictionary with keys 'res', 'bc', 'ic' indicating data type ranges
"""
def transform_data_for_snarenet(x, t):
    x_res, x_left, x_upper, x_lower = x
    t_res, t_left, t_upper, t_lower = t
    
    # Get device
    device = x_res.device if hasattr(x_res, 'device') else 'cpu'
    
    # Convert numpy arrays to torch tensors if needed, ensure requires_grad=True
    def to_tensor(arr):
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr, dtype=torch.float32, requires_grad=True).to(device)
        # If already a tensor, ensure it requires grad
        if isinstance(arr, torch.Tensor):
            if not arr.requires_grad:
                arr = arr.detach().requires_grad_(True)
            return arr
        return arr
    
    x_res = to_tensor(x_res)
    x_left = to_tensor(x_left)
    x_upper = to_tensor(x_upper)
    x_lower = to_tensor(x_lower)
    t_res = to_tensor(t_res)
    t_left = to_tensor(t_left)
    t_upper = to_tensor(t_upper)
    t_lower = to_tensor(t_lower)
    
    # Residual points: x1 = x2 = x_res (PDE enforcement at interior)
    x1_res = x_res
    x2_res = x_res.clone()
    t_res_paired = t_res
    
    # Boundary points: x1 = lower boundary, x2 = upper boundary (periodic BC)
    x1_bc = x_lower
    x2_bc = x_upper
    t_bc = t_lower  # Both should be same
    
    # Initial condition: x1 = x2 = x_left at t=0
    x1_ic = x_left
    x2_ic = x_left.clone()
    t_ic = t_left
    
    # Concatenate all data
    x1_all = torch.cat([x1_res, x1_bc, x1_ic], dim=0)
    x2_all = torch.cat([x2_res, x2_bc, x2_ic], dim=0)
    t_all = torch.cat([t_res_paired, t_bc, t_ic], dim=0)
    
    # Store indices for each data type
    n_res = x_res.shape[0]
    n_bc = x_lower.shape[0]
    n_ic = x_left.shape[0]
    
    indices = {
        'res': (0, n_res),
        'bc': (n_res, n_res + n_bc),
        'ic': (n_res + n_bc, n_res + n_bc + n_ic)
    }
    
    return x1_all, x2_all, t_all, indices

"""
Helper function for making predictions with SnareNetPINN model in PDE constraint mode.

INPUT:
- x1, x2: paired spatial coordinates
- t: temporal coordinate
- model: SnareNetPINN model (with pde_constraint set)
- indices: dictionary indicating data type ranges (from transform_data_for_snarenet)
OUTPUT:
- preds: tuple of (pred_res, pred_left, pred_upper, pred_lower)
         matching the format expected by loss functions
"""
def predict_snarenet(x1, x2, t, model, indices):
    # Get predictions for all data
    u1, u2 = model(x1, x2, t)  # Note: forward(x1, x2, t) signature
    
    # Split predictions based on data type
    res_start, res_end = indices['res']
    bc_start, bc_end = indices['bc']
    ic_start, ic_end = indices['ic']
    
    # Residual predictions (u1 = u2 since x1 = x2)
    pred_res = u1[res_start:res_end]
    
    # Initial condition predictions (u1 = u2 since x1 = x2)
    pred_left = u1[ic_start:ic_end]
    
    # Boundary predictions (u1 at lower boundary, u2 at upper boundary)
    pred_lower = u1[bc_start:bc_end]
    pred_upper = u2[bc_start:bc_end]
    
    preds = (pred_res, pred_left, pred_upper, pred_lower)
    
    return preds
