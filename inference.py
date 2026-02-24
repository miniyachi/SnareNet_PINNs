"""
Script to load a trained model checkpoint and perform inference/evaluation.

Usage (local checkpoint):
python3 inference.py --checkpoint checkpoints/snarenet_seed123_convection_beta_1.pt --device 0
python3 inference.py --checkpoint checkpoints/snarenet_seed123_convection_beta_1.pt --save_figure results.png

Usage (from wandb):
python3 inference.py --from_wandb --wandb_run_path entity/project/run_id --device 0
python3 inference.py --from_wandb --wandb_run_path entity/project/run_id --save_figure comparison.pdf

Usage (SnareNet comparison):
python3 inference.py --checkpoint checkpoints/snarenet.pt --save_figure comparison.png
python3 inference.py --checkpoint checkpoints/snarenet.pt --enable_projection --save_figure with_proj.png
"""

import argparse
import torch
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.models import PINN, SnareNetPINN
from src.train_utils import (
    get_data, get_pde, predict,
    get_ref_solutions,
    l1_relative_error, l2_relative_error, set_random_seed
)
from src.snarenet_utils import (
    get_pde_constraint, predict_snarenet, transform_data_for_snarenet
)

def reshape_solution(u, num_x, num_t):
    """Reshape flattened solution array to 2D grid for visualization"""
    u_res = u[:(num_t-1)*(num_x-2)].reshape(num_t-1, num_x-2)
    u_left = u[(num_t-1)*(num_x-2):(num_t-1)*(num_x-2)+num_x].flatten()
    u_upper = u[(num_t-1)*(num_x-2)+num_x:(num_t-1)*(num_x-2)+num_x+num_t].flatten()
    x_lower = u[(num_t-1)*(num_x-2)+num_x+num_t:].flatten()

    u_grid = np.zeros([num_t, num_x])
    u_grid[1:, 1:-1] = u_res
    u_grid[0, :] = u_left
    u_grid[:, 0] = x_lower
    u_grid[:, -1] = u_upper

    # change indexing from 'xy' to 'ij'
    u_grid = u_grid.T

    return u_grid


def plot_solution(fig, ax, u, num_x, num_t, x_range, t_range, title, vmin, vmax):
    """Plot a single solution on given axes"""
    pos = ax.imshow(
        reshape_solution(u, num_x, num_t),
        extent=np.concatenate([t_range, x_range]),
        cmap='viridis',
        aspect='auto',
        origin='lower',
        vmin=vmin,
        vmax=vmax
    )
    fig.colorbar(pos, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")


def plot_solution_comparison(target, pred_no_proj, pred_with_proj, num_x, num_t, x_range, t_range, 
                            pde_name, pde_params, save_path, is_snarenet=True):
    """Plot comparison of target and predicted solutions
    
    Args:
        target: Target solution array
        pred_no_proj: Prediction without projection (or main prediction for PINN)
        pred_with_proj: Prediction with projection (None for PINN)
        num_x, num_t: Grid dimensions
        x_range, t_range: Domain ranges
        pde_name: Name of PDE
        pde_params: PDE parameters
        save_path: Path to save figure
        is_snarenet: Whether model is SnareNet (shows 3 plots) or PINN (shows 2 plots)
    """
    # Set up matplotlib for LaTeX rendering (similar to notebook)
    mpl.rcParams.update({'font.size': 14})
    
    # Create figure with appropriate number of subplots
    if is_snarenet:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    # Create title based on PDE type
    if pde_name == "convection":
        beta = pde_params[1] if isinstance(pde_params, list) else pde_params
        title = f'Convection, β = {beta}'
    elif pde_name == "reaction":
        rho = pde_params[1] if isinstance(pde_params, list) else pde_params
        title = f'Reaction, ρ = {rho}'
    elif pde_name == "wave":
        beta = pde_params[1] if isinstance(pde_params, list) else pde_params
        title = f'Wave, β = {beta}'
    else:
        title = f'{pde_name.capitalize()}'
    
    fig.suptitle(title, fontsize=16)
    
    # Compute value range for consistent color scaling
    if is_snarenet:
        vmin = np.min([target, pred_no_proj, pred_with_proj])
        vmax = np.max([target, pred_no_proj, pred_with_proj])
    else:
        vmin = np.min([target, pred_no_proj])
        vmax = np.max([target, pred_no_proj])
    
    # Plot target solution
    plot_solution(fig, axes[0], target, num_x, num_t, x_range, t_range, 
                 'Target Solution', vmin, vmax)
    
    if is_snarenet:
        # Plot prediction without projection
        plot_solution(fig, axes[1], pred_no_proj, num_x, num_t, x_range, t_range, 
                     'Prediction (No Projection)', vmin, vmax)
        
        # Plot prediction with projection
        plot_solution(fig, axes[2], pred_with_proj, num_x, num_t, x_range, t_range, 
                     'Prediction (With Projection)', vmin, vmax)
    else:
        # Plot PINN prediction
        plot_solution(fig, axes[1], pred_no_proj, num_x, num_t, x_range, t_range, 
                     'PINN Prediction', vmin, vmax)
    
    # Save figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to: {save_path}")


def load_model_from_checkpoint(checkpoint_path, device='cpu', suppress_warnings=False):
    """Load model from checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        suppress_warnings: If True, suppress warnings about missing config (used when loading from W&B)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both old format (with model_config dict) and new format (flat)
    if 'model_config' in checkpoint:
        # Old format
        config = checkpoint['model_config']
        model_type = config['model_type']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
    else:
        # New simplified format
        model_type = checkpoint['model_type']
        hidden_dim = checkpoint['hidden_dim']
        num_layers = checkpoint['num_layers']
        config = checkpoint  # Use checkpoint directly as config
    
    print(f"\nModel configuration:")
    print(f"  Type: {model_type}")
    print(f"  Layers: {num_layers}")
    print(f"  Neurons: {hidden_dim}")
    
    # Create model based on type
    if model_type == 'pinn':
        model = PINN(
            in_dim=2,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layer=num_layers
        ).to(device)
    
    elif model_type == 'snarenet':
        # Recreate PDE constraint
        pde_constraint = get_pde_constraint(
            config['pde_name'],
            config['pde_coefs'],
            config['x_range'],
            config['t_range']
        )
        
        model = SnareNetPINN(
            in_dim=2,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layer=num_layers,
            pde_constraint=pde_constraint,
            newton_maxiter=config['newton_maxiter'],
            rtol=config['newton_rtol'],
            lambda_reg=config['lambda_reg'],
            if_project=config['if_project']
        ).to(device)
        
        print(f"  Projection: {config['if_project']}")
        print(f"  Newton max iter: {config['newton_maxiter']}")
        print(f"  Lambda reg: {config['lambda_reg']}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nCheckpoint loaded successfully!")
    
    # Build exp_args from checkpoint for evaluation
    # Handle both old and new formats
    if 'experiment_args' in checkpoint:
        exp_args = checkpoint['experiment_args']
    else:
        # Reconstruct minimal exp_args from checkpoint
        pde_name = config.get('pde_name', 'convection')
        
        # Reconstruct pde_params from pde_coefs
        pde_params = None
        if 'pde_coefs' in config and config['pde_coefs'] is not None:
            pde_coefs = config['pde_coefs']
            if pde_name == 'convection':
                # pde_coefs can be dict {'beta': value} or list [beta]
                if isinstance(pde_coefs, dict):
                    pde_params = ['beta', str(pde_coefs.get('beta', 1))]
                else:
                    pde_params = ['beta', str(pde_coefs[0])]
            elif pde_name == 'reaction':
                # pde_coefs can be dict {'rho': value} or list [rho]
                if isinstance(pde_coefs, dict):
                    pde_params = ['rho', str(pde_coefs.get('rho', 1))]
                else:
                    pde_params = ['rho', str(pde_coefs[0])]
            elif pde_name == 'wave':
                # pde_coefs can be dict {'beta': value} or list [beta]
                if isinstance(pde_coefs, dict):
                    pde_params = ['beta', str(pde_coefs.get('beta', 1))]
                else:
                    pde_params = ['beta', str(pde_coefs[0])]
        
        exp_args = {
            'model': model_type,
            'num_layers': num_layers,
            'num_neurons': hidden_dim,
            'pde': pde_name,
            'pde_params': pde_params,
            'loss': 'mse',  # Default
            'data_loss_weight': 0.0,  # Default
            'num_data_samples': None,  # Default
            'num_x': 257,  # Default
            'num_t': 101,  # Default  
            'num_res': 10000  # Default
        }
        if not suppress_warnings:
            print("\n⚠️  Using default values for missing experiment configuration.")
            print("   For full configuration, load from W&B using --from_wandb")
    
    return model, config, exp_args


def load_model_from_wandb(wandb_run_path, checkpoint_name=None, device='cpu', download_dir='./wandb_checkpoints'):
    """
    Load model from W&B run.
    
    Args:
        wandb_run_path: W&B run path in format 'entity/project/run_id' or full URL
        checkpoint_name: Specific checkpoint file to load (e.g., 'model.pt'). 
                        If None, will try to find checkpoint automatically.
        device: Device to load model on
        download_dir: Directory to download checkpoint to
        
    Returns:
        model, config, exp_args (same as load_model_from_checkpoint)
    """
    print(f"Loading checkpoint from W&B run: {wandb_run_path}")
    
    # Parse run path if it's a URL
    if 'wandb.ai' in wandb_run_path:
        # Extract entity/project/run_id from URL
        # Format: https://wandb.ai/entity/project/runs/run_id
        parts = wandb_run_path.split('/')
        if 'runs' in parts:
            run_idx = parts.index('runs')
            entity = parts[run_idx - 2]
            project = parts[run_idx - 1]
            run_id = parts[run_idx + 1].split('?')[0]  # Remove query params
            wandb_run_path = f"{entity}/{project}/{run_id}"
            print(f"Parsed run path: {wandb_run_path}")
    
    # Initialize W&B API
    api = wandb.Api()
    
    try:
        run = api.run(wandb_run_path)
        print(f"Found run: {run.name}")
        print(f"  Project: {run.project}")
        print(f"  Created: {run.created_at}")
    except Exception as e:
        print(f"Error accessing W&B run: {e}")
        print(f"Make sure the run path is correct: 'entity/project/run_id'")
        raise
    
    # List available files
    files = run.files()
    checkpoint_files = [f for f in files if f.name.endswith('.pt')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (.pt) found in run {wandb_run_path}")
    
    print(f"\nAvailable checkpoints:")
    for i, f in enumerate(checkpoint_files):
        print(f"  [{i}] {f.name} ({f.size / 1024 / 1024:.2f} MB)")
    
    # Select checkpoint
    if checkpoint_name is None:
        if len(checkpoint_files) == 1:
            checkpoint_file = checkpoint_files[0]
            print(f"\nAutomatically selected: {checkpoint_file.name}")
        else:
            # Try to find checkpoint in checkpoints/ directory
            checkpoint_dir_files = [f for f in checkpoint_files if f.name.startswith('checkpoints/')]
            if checkpoint_dir_files:
                checkpoint_file = checkpoint_dir_files[0]
                print(f"\nAutomatically selected: {checkpoint_file.name}")
            else:
                checkpoint_file = checkpoint_files[0]
                print(f"\nMultiple checkpoints found. Using first one: {checkpoint_file.name}")
                print(f"To specify a different checkpoint, use --checkpoint_name")
    else:
        # Find matching checkpoint
        matching = [f for f in checkpoint_files if checkpoint_name in f.name]
        if not matching:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found. Available: {[f.name for f in checkpoint_files]}")
        checkpoint_file = matching[0]
        print(f"\nSelected checkpoint: {checkpoint_file.name}")
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    
    # Download checkpoint
    local_path = os.path.join(download_dir, os.path.basename(checkpoint_file.name))
    print(f"Downloading to: {local_path}")
    checkpoint_file.download(root=download_dir, replace=True)
    
    # If file was in a subdirectory, move it to download_dir root
    downloaded_path = os.path.join(download_dir, checkpoint_file.name)
    if downloaded_path != local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(downloaded_path):
            import shutil
            shutil.move(downloaded_path, local_path)
    
    print(f"Download complete!")
    
    # Load checkpoint using existing function (suppress warning since we'll load from W&B)
    model, config, exp_args = load_model_from_checkpoint(local_path, device, suppress_warnings=True)
    
    # Enhance exp_args with W&B run config if available
    try:
        print(f"\nRetrieving full experiment configuration from W&B...")
        wandb_config = dict(run.config)
        if wandb_config:
            # Override with full config from W&B
            exp_args = wandb_config
            print(f"✓ Loaded full configuration from W&B run")
    except Exception as e:
        print(f"⚠️  Could not retrieve W&B config: {e}")
        print(f"   Using configuration from checkpoint file")
    
    return model, config, exp_args


def evaluate_model(model, config, exp_args, device='cpu', enable_projection=None):
    """Evaluate model on train and test sets"""
    
    is_snarenet = (config['model_type'] == 'snarenet')
    
    # Override projection setting if specified
    if is_snarenet and enable_projection is not None:
        model.set_projection(enable_projection)
        print(f"\nProjection manually set to: {enable_projection}")
    
    # Get data_loss_weight from experiment config (default to 0.0 for backward compatibility)
    data_loss_weight = exp_args['data_loss_weight']
    
    # Get PDE information
    x_range, t_range, loss_func, pde_coefs = get_pde(
        exp_args["pde"],
        exp_args["pde_params"],
        exp_args["loss"]
    )
    
    # Compute target solutions for data loss if needed
    target_solutions = None
    if data_loss_weight > 0:
        print(f"Data loss weight: {data_loss_weight}")
    
    # Generate training data (same as training)
    x, t, data_params = get_data(
        x_range, t_range,
        exp_args["num_x"],
        exp_args["num_t"],
        random=True,
        num_res_samples=exp_args["num_res"],
        device=device
    )
    
    # Compute target solutions if data loss is used
    if data_loss_weight > 0:
        target_solutions_np = get_ref_solutions(exp_args["pde"], pde_coefs, x, t, data_params)
        target_solutions = torch.tensor(target_solutions_np, dtype=torch.float32, device=device)
    
    # Generate test data (coarse grid)
    n_x_test = int((exp_args["num_x"] - 1) / 2) + 1
    n_t_test = exp_args["num_t"]
    x_test, t_test, data_params_test = get_data(
        x_range, t_range,
        n_x_test, n_t_test,
        random=False,
        device=device
    )
    
    print(f"\n=== Evaluation ===")
    print(f"PDE: {exp_args['pde']}")
    print(f"PDE params: {exp_args['pde_params']}")
    print(f"Train points: {exp_args['num_x']} x {exp_args['num_t']} (res: {exp_args['num_res']})")
    print(f"Test points: {n_x_test} x {n_t_test}")
    
    # Evaluate on training set
    if is_snarenet:
        x1, x2, t_paired, indices = transform_data_for_snarenet(x, t)
        # SnareNet needs gradients for forward pass (computes derivatives)
        preds = predict_snarenet(x1, x2, t_paired, model, indices)
        predictions = torch.vstack(preds).cpu().detach().numpy()
        
        # Compute PDE losses
        loss_res, loss_bc, loss_ic, loss_data = loss_func(x, t, preds, target_solutions, data_indices=None)
        print(f"\n=== Training Set Metrics ===")
        print(f"Projection enabled: {model._if_project}")
        print(f"Loss (residual): {loss_res.item():.6e}")
        print(f"Loss (boundary): {loss_bc.item():.6e}")
        print(f"Loss (initial):  {loss_ic.item():.6e}")
        if data_loss_weight > 0:
            print(f"Loss (data):     {loss_data.item():.6e} (weight: {data_loss_weight})")
        print(f"Total loss:      {(loss_res + loss_bc + loss_ic + data_loss_weight * loss_data).item():.6e}")
    else:
        # PINN also needs gradients for loss computation (PDE residuals)
        preds = predict(x, t, model)
        predictions = torch.vstack(preds).cpu().detach().numpy()
        
        loss_res, loss_bc, loss_ic, loss_data = loss_func(x, t, preds, target_solutions, data_indices=None)
        print(f"\n=== Training Set Metrics ===")
        print(f"Loss (residual): {loss_res.item():.6e}")
        print(f"Loss (boundary): {loss_bc.item():.6e}")
        print(f"Loss (initial):  {loss_ic.item():.6e}")
        if data_loss_weight > 0:
            print(f"Loss (data):     {loss_data.item():.6e} (weight: {data_loss_weight})")
        print(f"Total loss:      {(loss_res + loss_bc + loss_ic + data_loss_weight * loss_data).item():.6e}")
    
    targets = get_ref_solutions(exp_args["pde"], pde_coefs, x, t, data_params)
    train_l1re = l1_relative_error(predictions, targets)
    train_l2re = l2_relative_error(predictions, targets)
    
    print(f"L1 Relative Error: {train_l1re:.6e}")
    print(f"L2 Relative Error: {train_l2re:.6e}")
    
    # Evaluate on test set
    if is_snarenet:
        x1_test, x2_test, t_test_paired, indices_test = transform_data_for_snarenet(x_test, t_test)
        # SnareNet needs gradients for forward pass
        preds_test = predict_snarenet(x1_test, x2_test, t_test_paired, model, indices_test)
        predictions_test = torch.vstack(preds_test).cpu().detach().numpy()
    else:
        # PINN: no gradients needed for test set (no loss computation)
        with torch.no_grad():
            preds_test = predict(x_test, t_test, model)
            predictions_test = torch.vstack(preds_test).cpu().detach().numpy()
    
    targets_test = get_ref_solutions(exp_args["pde"], pde_coefs, x_test, t_test, data_params_test)
    test_l1re = l1_relative_error(predictions_test, targets_test)
    test_l2re = l2_relative_error(predictions_test, targets_test)
    
    print(f"\n=== Test Set Metrics ===")
    print(f"L1 Relative Error: {test_l1re:.6e}")
    print(f"L2 Relative Error: {test_l2re:.6e}")
    
    results = {
        'train': {
            'l1re': train_l1re,
            'l2re': train_l2re,
            'loss_res': loss_res.item(),
            'loss_bc': loss_bc.item(),
            'loss_ic': loss_ic.item(),
            'loss_data': loss_data.item(),
            'total_loss': (loss_res + loss_bc + loss_ic + data_loss_weight * loss_data).item()
        },
        'test': {
            'l1re': test_l1re,
            'l2re': test_l2re
        }
    }
    
    return results, predictions, predictions_test, targets, targets_test


def main():
    parser = argparse.ArgumentParser(
        description='Load and evaluate a trained model checkpoint',
        epilog="""
Examples:
  # Load from local file and save visualization
  python3 inference.py --checkpoint checkpoints/model.pt --device 0 --save_figure results.png
  
  # Load from W&B and save comparison (SnareNet: shows 3 plots)
  python3 inference.py --from_wandb --wandb_run_path entity/project/run_id --save_figure comparison.pdf
  
  # Load specific checkpoint from W&B
  python3 inference.py --from_wandb --wandb_run_path entity/project/run_id --checkpoint_name model.pt
  
  # SnareNet: evaluate only with projection enabled
  python3 inference.py --checkpoint snarenet.pt --enable_projection --save_figure with_proj.png
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Loading method
    parser.add_argument('--from_wandb', action='store_true',
                        help='Load checkpoint from W&B instead of local file')
    
    # Local checkpoint options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to local model checkpoint (.pt file) - required if not using --from_wandb')
    
    # W&B options
    parser.add_argument('--wandb_run_path', type=str, default=None,
                        help='W&B run path (entity/project/run_id) or full URL - required if using --from_wandb')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='[W&B only] Specific checkpoint filename to load (e.g., "model.pt"). If not specified, will auto-select.')
    parser.add_argument('--download_dir', type=str, default='./wandb_checkpoints',
                        help='[W&B only] Directory to download checkpoint to (default: ./wandb_checkpoints)')
    
    # Evaluation options
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda device number)')
    parser.add_argument('--enable_projection', action='store_true',
                        help='[SnareNet only] Enable projection during evaluation')
    parser.add_argument('--disable_projection', action='store_true',
                        help='[SnareNet only] Disable projection during evaluation')
    parser.add_argument('--save_figure', type=str, default=None,
                        help='Path to save visualization figure (optional, .png or .pdf format)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.from_wandb and args.wandb_run_path is None:
        parser.error("--wandb_run_path is required when using --from_wandb")
    if not args.from_wandb and args.checkpoint is None:
        parser.error("--checkpoint is required when not using --from_wandb")
    if args.checkpoint_name and not args.from_wandb:
        parser.error("--checkpoint_name can only be used with --from_wandb")
    
    # Determine device
    if args.device.lower() == 'cpu':
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = f'cuda:{args.device}'
        else:
            print("CUDA not available, using CPU")
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model based on method
    if args.from_wandb:
        print(f"\n{'='*60}")
        print("Loading from Weights & Biases")
        print(f"{'='*60}")
        model, config, exp_args = load_model_from_wandb(
            wandb_run_path=args.wandb_run_path,
            checkpoint_name=args.checkpoint_name,
            device=device,
            download_dir=args.download_dir
        )
    else:
        print(f"\n{'='*60}")
        print("Loading from local checkpoint")
        print(f"{'='*60}")
        model, config, exp_args = load_model_from_checkpoint(args.checkpoint, device)
    
    # Get data_loss_weight from experiment config (default to 0.0 for backward compatibility)
    data_loss_weight = exp_args.get('data_loss_weight', 0.0)
    
    # Get PDE information for plotting
    x_range, t_range, _, _ = get_pde(
        exp_args["pde"],
        exp_args["pde_params"],
        exp_args["loss"]
    )
    
    # For SnareNet, evaluate with both projection settings; for PINN, evaluate once
    is_snarenet = (config['model_type'] == 'snarenet')
    
    if is_snarenet:
        # Check if user specified a single projection setting
        if args.enable_projection and args.disable_projection:
            raise ValueError("Cannot specify both --enable_projection and --disable_projection")
        
        if args.enable_projection:
            # Only evaluate with projection
            print("\n" + "="*60)
            print("Evaluating with projection enabled")
            print("="*60)
            results_proj, train_preds_proj, test_preds_proj, train_targets, test_targets = evaluate_model(
                model, config, exp_args, device, enable_projection=True
            )
            results = results_proj
            test_preds_no_proj = None
            test_preds = test_preds_proj
        elif args.disable_projection:
            # Only evaluate without projection
            print("\n" + "="*60)
            print("Evaluating with projection disabled")
            print("="*60)
            results_no_proj, train_preds_no_proj, test_preds_no_proj, train_targets, test_targets = evaluate_model(
                model, config, exp_args, device, enable_projection=False
            )
            results = results_no_proj
            test_preds = test_preds_no_proj
            test_preds_proj = None
        else:
            # Evaluate both ways for comparison
            print("\n" + "="*60)
            print("Evaluating without projection")
            print("="*60)
            results_no_proj, train_preds_no_proj, test_preds_no_proj, train_targets, test_targets = evaluate_model(
                model, config, exp_args, device, enable_projection=False
            )
            
            print("\n" + "="*60)
            print("Evaluating with projection")
            print("="*60)
            results_proj, train_preds_proj, test_preds_proj, _, _ = evaluate_model(
                model, config, exp_args, device, enable_projection=True
            )
            
            results = {'no_projection': results_no_proj, 'with_projection': results_proj}
            test_preds = test_preds_proj  # Use projection version as primary
    else:
        # PINN model - single evaluation
        enable_projection = None
        if args.enable_projection or args.disable_projection:
            print("⚠️  Projection flags are ignored for PINN models")
        
        results, train_preds, test_preds, train_targets, test_targets = evaluate_model(
            model, config, exp_args, device, enable_projection
        )
        test_preds_no_proj = test_preds
        test_preds_proj = None
    
    # Save figure if requested
    if args.save_figure:
        n_x_test = int((exp_args["num_x"] - 1) / 2) + 1
        n_t_test = exp_args["num_t"]
        
        plot_solution_comparison(
            target=test_targets,
            pred_no_proj=test_preds_no_proj if is_snarenet else test_preds,
            pred_with_proj=test_preds_proj if is_snarenet else None,
            num_x=n_x_test,
            num_t=n_t_test,
            x_range=x_range,
            t_range=t_range,
            pde_name=exp_args["pde"],
            pde_params=exp_args["pde_params"],
            save_path=args.save_figure,
            is_snarenet=is_snarenet
        )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
