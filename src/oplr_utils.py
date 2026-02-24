import numpy as np
import torch
import torch.nn as nn
import deepxde as dde
import wandb

from .train_utils import get_data, parse_params_list, get_opt_params, l1_relative_error, l2_relative_error


def set_float_precision(precision):
    """
    Helper function for setting floating point precision for PyTorch and DeepXDE.

    INPUT: 
    - precision: integer (32 or 64); floating point precision
    """
    if precision == 32:
        torch.set_default_dtype(torch.float32)
        dde.config.set_default_float("float32")
    elif precision == 64:
        torch.set_default_dtype(torch.float64)
        dde.config.set_default_float("float64")
    else:
        raise ValueError(f"Precision must be 32 or 64, got {precision}")


def get_data_oplr(func_space, x_range, t_range, n_funcs, n_x, n_t, n_res, device='cpu'):
    """
    Sample GRF functions and generate collocation points for operator learning.

    Returns:
        funcs: NumPy array of shape (n_funcs, n_features)
        x: tuple of (x_res, x_left, x_upper, x_lower) tensors
        t: tuple of (t_res, t_left, t_upper, t_lower) tensors
        data_params: dict with grid metadata
    """
    funcs = func_space.random(size=n_funcs)
    x, t, data_params = get_data(x_range, t_range, n_x, n_t, random=True, num_res_samples=n_res, device=device)

    x_res, x_left, x_upper, x_lower = x
    t_res, t_left, t_upper, t_lower = t
    x = (x_res, x_left, x_upper, x_lower)
    t = (t_res, t_left, t_upper, t_lower)
    return funcs, x, t, data_params


def get_func_rep(func_space, funcs, sensor_x, device='cpu'):
    """
    Evaluate each function at the sensor points to form branch inputs.

    Args:
        func_space: GRF space object
        funcs: NumPy array of shape (n_funcs, n_features)
        sensor_x: tensor of shape (n_sensor, 1) — sensor positions (endpoint=False)

    Returns:
        branch_inputs: tensor of shape (n_funcs, n_sensor)
    """
    branch = func_space.eval_batch(funcs, sensor_x.cpu().detach().numpy().astype(np.float64))  # (n_funcs, n_sensor)
    return torch.tensor(branch, dtype=torch.get_default_dtype()).to(device)


def get_ref_solutions_oplr(pde_name, pde_coefs, x, t, data_params, func_space, funcs):
    """
    Compute reference solutions for the operator learning PDE family.
    For convection: u_i(x, t) = f_i(x - beta * t) with periodic wrapping.

    Returns:
        sol: NumPy array of shape (n_funcs, N_total) where columns follow [res, left, upper, lower] ordering
    """
    if pde_name == "convection":
        parts = []
        for xi, ti in zip(x, t):
            shifted = (xi - pde_coefs["beta"] * ti) % (2 * np.pi)       # tensor arithmetic
            shifted = torch.clamp(shifted, func_space.x.min(), func_space.x.max())  # guard boundary edge cases
            # eval_batch requires float64 numpy array input
            vals = func_space.eval_batch(funcs, shifted.cpu().detach().numpy().astype(np.float64))  # (n_funcs, N_part)
            parts.append(vals)
        sol = np.concatenate(parts, axis=1)  # (n_funcs, N_total)
    else:
        raise RuntimeError(f"{pde_name} is not supported for operator reference solution.")
    return sol


def get_pde_oplr(pde_name, pde_params_list, loss_name, funcspace_params_list=None, n_sensor=100, device='cpu'):
    """
    Build the GRF function space and physics loss for operator learning.

    Args:
        pde_name (str): PDE identifier (e.g. 'convection').
        pde_params_list (list[str]): Raw PDE parameter list, e.g. ['beta', '1.0'].
        loss_name (str): Loss type — one of 'l1', 'mse', 'huber', 'hybrid'.
        funcspace_params_list (list[str]): Raw GRF parameter list,
            e.g. ['length_scale', '0.2', 'N', '1000'].
            Required keys: length_scale (float), N (int).
        n_sensor (int): Number of branch sensor points.
        device (str): Device for tensor allocation.

    Returns:
        func_space: GRF function space built from funcspace_params_list and the PDE domain.
        sensor_x: tensor of shape (n_sensor, 1) — sensor positions for branch input evaluation.
        x_range (list[float]): Spatial domain [x_min, x_max].
        t_range (list[float]): Temporal domain [t_min, t_max].
        loss_func (callable): Physics loss with signature
            loss_func(func_space, funcs, x, t, pred) -> (loss_res, loss_bc, loss_ic),
            where pred = (pred_res, pred_left, pred_upper, pred_lower),
            each of shape (n_funcs, N_seg). IC targets are evaluated from func_space
            at x_left (IC collocation points) inside loss_func.
        pde_coefs (dict): Parsed PDE coefficients, e.g. {'beta': 1.0}.
    """
    loss_options = {
        "l1":     {"res": nn.L1Loss(),    "bc": nn.L1Loss(),    "ic": nn.L1Loss()},
        "mse":    {"res": nn.MSELoss(),   "bc": nn.MSELoss(),   "ic": nn.MSELoss()},
        "huber":  {"res": nn.HuberLoss(), "bc": nn.HuberLoss(), "ic": nn.HuberLoss()},
        "hybrid": {"res": nn.HuberLoss(), "bc": nn.MSELoss(),   "ic": nn.MSELoss()},
    }
    loss_type = loss_options[loss_name]

    pde_coefs  = parse_params_list(pde_params_list)
    funcspace_params = parse_params_list(funcspace_params_list)

    if "length_scale" not in funcspace_params:
        raise KeyError("length_scale is not specified for GRF function space.")

    if pde_name == "convection":
        if "beta" not in pde_coefs:
            raise KeyError("beta is not specified for convection PDE.")

        x_range = [0, 2 * np.pi]
        t_range = [0, 1]
        # Match GRF precision to torch default dtype set by set_float_precision
        torch_to_numpy_dtype = {torch.float32: np.float32, torch.float64: np.float64}
        np_dtype = torch_to_numpy_dtype[torch.get_default_dtype()]
        func_space = dde.data.GRF(
            T=np_dtype(x_range[1]),   # Use matching dtype so interp range matches tensor precision
            length_scale=funcspace_params['length_scale'],
        )

        def loss_func(func_space, funcs, x, t, pred):
            x_res, x_left, x_upper, x_lower = x
            t_res, t_left, t_upper, t_lower = t
            pred_res, pred_left, pred_upper, pred_lower = pred
            # Evaluate GRF at x_left (IC training points) — shapes match pred_left naturally
            # Use current default dtype so GRF interp range matches tensor precision
            ic_targets_np = func_space.eval_batch(funcs, x_left.cpu().detach().numpy())  # (n_funcs, n_x)
            ic_targets = torch.tensor(ic_targets_np, dtype=torch.get_default_dtype()).to(x_res.device)
            # pred_res: (n_funcs, n_res); x_res: (n_res, 1)
            u_x = torch.autograd.grad(
                pred_res, x_res, grad_outputs=torch.ones_like(pred_res),
                retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(
                pred_res, t_res, grad_outputs=torch.ones_like(pred_res),
                retain_graph=True, create_graph=True)[0]
            loss_res = loss_type["res"](u_t + pde_coefs["beta"] * u_x, torch.zeros_like(u_t))
            loss_bc  = loss_type["bc"](pred_upper - pred_lower, torch.zeros_like(pred_upper))
            loss_ic  = loss_type["ic"](pred_left, ic_targets)
            return loss_res, loss_bc, loss_ic

    else:
        raise RuntimeError(f"{pde_name} is not a supported PDE for operator learning.")

    # Sensor positions — use current default dtype, consistent with all other x/t coordinates
    sensor_x = torch.tensor(
        np.linspace(x_range[0], x_range[1], n_sensor, endpoint=False).reshape(-1, 1),
        dtype=torch.get_default_dtype(), requires_grad=True).to(device)

    return func_space, sensor_x, x_range, t_range, loss_func, pde_coefs


def get_model_oplr(model_type, n_sensors, branch_hidden, trunk_hidden, activation="sin", kernel_init="Glorot normal"):
    """
    Build a neural operator model.

    Args:
        model_type (str): Neural operator architecture.
        n_sensors (int): Number of branch input sensors
        branch_hidden (list[int]): Hidden layer sizes for branch network
        trunk_hidden (list[int]): Hidden layer sizes for trunk network
        activation (str): Activation function name
        kernel_init (str): Kernel initializer name

    Returns:
        model: Neural operator instance
    """
    branch_layers = [n_sensors] + branch_hidden
    trunk_layers  = [2] + trunk_hidden

    if model_type == 'DeepONetCartesianProd':
        return dde.nn.DeepONetCartesianProd(
            branch_layers, trunk_layers,
            activation=activation,
            kernel_initializer=kernel_init
        )
    else:
        raise ValueError(f"Model type '{model_type}' is not implemented.")


def predict_oplr(branch_inputs, x, t, model):
    """
    Forward pass through DeepONet for all functions and all collocation points.

    Args:
        branch_inputs: tensor (n_funcs, n_sensors)
        x: tuple (x_res, x_left, x_upper, x_lower)
        t: tuple (t_res, t_left, t_upper, t_lower)
        model: DeepONetCartesianProd

    Returns:
        (pred_res, pred_left, pred_upper, pred_lower), each (n_funcs, N_seg)
    """
    x_res, x_left, x_upper, x_lower = x
    t_res, t_left, t_upper, t_lower = t

    pred_res   = model([branch_inputs, torch.cat([x_res,   t_res],   dim=1)])
    pred_left  = model([branch_inputs, torch.cat([x_left,  t_left],  dim=1)])
    pred_upper = model([branch_inputs, torch.cat([x_upper, t_upper], dim=1)])
    pred_lower = model([branch_inputs, torch.cat([x_lower, t_lower], dim=1)])

    preds = (pred_res, pred_left, pred_upper, pred_lower)

    return preds


def train_oplr(model,
               pde_name,
               pde_params,
               loss_name,
               opt_params_list,
               funcspace_params_list,
               n_funcs,
               n_sensor,
               n_x,
               n_t,
               n_res,
               num_epochs,
               device,
               model_type="DeepONetCartesianProd",
               log_freq=20):
    """
    Train a neural operator with physics-based (unsupervised) loss.
    Mirrors train() from train_utils.py but for DeepONet.
    """
    func_space, sensor_x, x_range, t_range, loss_func, pde_coefs = get_pde_oplr(pde_name, pde_params, loss_name, funcspace_params_list, n_sensor, device)
    opt_params = get_opt_params(opt_params_list)
    opt = torch.optim.Adam(model.parameters(), **opt_params)

    # Generate data (fixed for all epochs)
    funcs, x, t, data_params = get_data_oplr(func_space, x_range, t_range, n_funcs, n_x, n_t, n_res, device)
    branch_inputs = get_func_rep(func_space, funcs, sensor_x, device)  # (n_funcs, n_sensors)

    # Initial evaluation before training
    model.eval()
    preds = predict_oplr(branch_inputs, x, t, model)
    loss_res, loss_bc, loss_ic = loss_func(func_space, funcs, x, t, preds)
    loss = loss_res + loss_bc + loss_ic
    wandb.log({
        'train/loss': loss.item(), 
        'train/loss_res': loss_res.item(),
        'train/loss_bc': loss_bc.item(), 
        'train/loss_ic': loss_ic.item()})

    for i in range(num_epochs):
        model.train()
        opt.zero_grad()
        preds = predict_oplr(branch_inputs, x, t, model)
        loss_res, loss_bc, loss_ic = loss_func(func_space, funcs, x, t, preds)
        loss = loss_res + loss_bc + loss_ic
        loss.backward()
        opt.step()

        if i % log_freq == 0:
            log_dict = {
                'train/loss':     loss.item(),
                'train/loss_res': loss_res.item(),
                'train/loss_bc':  loss_bc.item(),
                'train/loss_ic':  loss_ic.item(),
            }
            # L1/L2 error against exact solution
            model.eval()
            with torch.no_grad():
                preds_eval = predict_oplr(branch_inputs, x, t, model)
            pred_all = torch.cat(preds_eval, dim=1).cpu().detach().numpy()  # (n_funcs, N_total)
            sol = get_ref_solutions_oplr(pde_name, pde_coefs, x, t, data_params, func_space, funcs)
            log_dict['train/l1_re'] = l1_relative_error(pred_all, sol)
            log_dict['train/l2_re'] = l2_relative_error(pred_all, sol)
            wandb.log(log_dict)

    # Evaluate final train error
    model.eval()
    with torch.no_grad():
        preds_eval = predict_oplr(branch_inputs, x, t, model)
    pred_all = torch.cat(preds_eval, dim=1).cpu().detach().numpy()  # (n_funcs, N_total)
    sol = get_ref_solutions_oplr(pde_name, pde_coefs, x, t, data_params, func_space, funcs)
    train_l1re = l1_relative_error(pred_all, sol)
    train_l2re = l2_relative_error(pred_all, sol)

    # Coarse grid ane newly-sampled functions for testing (same sensor_x for branch input)
    n_funcs_test = n_funcs // 10
    n_x_test = int((n_x - 1) / 2) + 1
    n_t_test = n_t
    funcs_test, x_test, t_test, data_params_test = get_data_oplr(
        func_space, x_range, t_range, n_funcs_test, n_x_test, n_t_test, n_res, device)
    branch_inputs_test = get_func_rep(func_space, funcs_test, sensor_x, device)

    model.eval()
    with torch.no_grad():
        preds_test = predict_oplr(branch_inputs_test, x_test, t_test, model)
    pred_all_test = torch.cat(preds_test, dim=1).cpu().detach().numpy()  # (n_funcs, N_total_test)
    sol_test = get_ref_solutions_oplr(pde_name, pde_coefs, x_test, t_test, data_params_test, func_space, funcs_test)
    test_l1re = l1_relative_error(pred_all_test, sol_test)
    test_l2re = l2_relative_error(pred_all_test, sol_test)

    wandb.log({'final/train_l1re': train_l1re,
               'final/train_l2re': train_l2re,
               'final/test_l1re':  test_l1re,
               'final/test_l2re':  test_l2re})
