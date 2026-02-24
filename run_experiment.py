## Usage:
# Standard PINN:
# python3 run_experiment.py --seed 123 --pde convection --pde_params beta 1 --opt adam --opt_params lr 0.001 
# --num_layers 4 --num_neurons 50 --loss mse --num_x 257 --num_t 101 --num_res 10000 --epochs 4000 --wandb_project test_pinns --device 0

# SnareNet PINN:
# python3 run_experiment.py --model snarenet --seed 123 --pde convection --pde_params beta 40 --opt adam --opt_params lr 0.001 \
# --num_layers 4 --num_neurons 50 --loss mse --num_x 257 --num_t 101 --num_res 10000 --epochs 4000 \
# --newton_maxiter 10 --newton_rtol 1e-3 --lambda_reg 0.01 --wandb_project test_snarenet --device 0

# SnareNet with soft training (disable projection for first 1000 epochs):
# python3 run_experiment.py --model snarenet --seed 123 --pde convection --pde_params beta 40 --opt adam --opt_params lr 0.001 \
# --num_layers 4 --num_neurons 50 --loss mse --num_x 257 --num_t 101 --num_res 10000 --epochs 4000 \
# --newton_maxiter 10 --newton_rtol 1e-3 --lambda_reg 0.01 --soft_epochs 1000 --wandb_project test_snarenet --device 0

# external libraries and packages
import wandb
import argparse
import os
import sys
import traceback
import torch

from src.train_utils import set_random_seed, train, get_pde
from src.snarenet_utils import get_pde_constraint
from src.models import PINN, SnareNetPINN

def save_checkpoint(model, experiment_args, wandb_name, pde_coefs=None, x_range=None, t_range=None):
    """
    Save model checkpoint with minimal configuration (weights + architecture).
    
    Args:
        model: The trained model
        experiment_args: Dictionary of experiment configuration (already saved in W&B config)
        wandb_name: Name for the checkpoint file
        pde_coefs: PDE coefficients (for SnareNet only)
        x_range: Spatial domain range (for SnareNet only)
        t_range: Temporal domain range (for SnareNet only)
    """
    # Save model checkpoint (minimal - just weights and architecture config)
    # Note: experiment_args is already saved in W&B config, no need to duplicate
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': experiment_args["model"],
        'hidden_dim': experiment_args["num_neurons"],
        'num_layers': experiment_args["num_layers"]
    }
    
    # Add SnareNet-specific config if applicable (needed to recreate PDE constraint)
    if experiment_args["model"] == "snarenet":
        checkpoint.update({
            'pde_name': experiment_args["pde"],
            'pde_coefs': pde_coefs,
            'x_range': x_range,
            't_range': t_range,
            'newton_maxiter': experiment_args["newton_maxiter"],
            'newton_rtol': experiment_args["newton_rtol"],
            'lambda_reg': experiment_args["lambda_reg"],
            'if_project': experiment_args["if_project"]
        })
    
    checkpoint_path = f"checkpoints/{wandb_name}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")
    
    # Upload checkpoint to wandb
    wandb.save(checkpoint_path)
    print(f"Model checkpoint uploaded to W&B")

def set_wandb_run_name(model, seed, pde_name, pde_params):
    """
    Generate a wandb run name in the format: model_seed{seed}_{pde_name}_{pde_params}
    
    Args:
        model (str): Model type ('pinn' or 'snarenet')
        seed (int): Random seed for the experiment
        pde_name (str): Name of the PDE (e.g., 'convection', 'reaction', 'wave')
        pde_params (list or None): List of PDE parameters as strings
    
    Returns:
        str: Formatted run name
    """
    if pde_params is None:
        params_str = "default"
    else:
        params_str = "_".join(pde_params)
    
    return f"{model}_seed{seed}_{pde_name}_{params_str}"

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pinn', 
                        choices=['pinn', 'snarenet'],
                        help='model type: pinn (standard) or snarenet (hard constraints)')
    parser.add_argument('--seed', type=int, default=1234, help='initial seed')
    parser.add_argument('--pde', type=str,
                        default='convection', help='PDE type')
    parser.add_argument('--pde_params', nargs='+', type=str,
                        default=None, help='PDE coefficients')
    parser.add_argument('--opt', type=str, default='lbfgs',
                        help='optimizer to use')
    parser.add_argument('--opt_params', nargs='+', type=str,
                        default=None, help='optimizer parameters')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers of the neural net')
    parser.add_argument('--num_neurons', type=int, default=50,
                        help='number of neurons per layer')
    parser.add_argument('--loss', type=str, default='mse',
                        help='type of loss function')
    parser.add_argument('--data_loss_weight', type=float, default=0.0,
                        help='weight for data loss (0=disabled, >0=enabled). Data loss measures error to reference solution.')
    parser.add_argument('--num_data_samples', type=int, default=None,
                        help='number of randomly sampled data points to use for data loss (None=use all available points)')
    parser.add_argument('--num_x', type=int, default=257,
                        help='number of spatial sample points (power of 2 + 1)')
    parser.add_argument('--num_t', type=int, default=101,
                        help='number of temporal sample points')
    parser.add_argument('--num_res', type=int, default=10000,
                        help='number of sampled residual points')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to run')
    parser.add_argument('--wandb_project', type=str,
                        default='pinns', help='W&B project name')
    parser.add_argument('--device', type=str, default=0, help='GPU to use')
    
    # SnareNet-specific parameters (only used when --model snarenet)
    parser.add_argument('--newton_maxiter', type=int, default=10,
                        help='[SnareNet] max Newton iterations for projection')
    parser.add_argument('--newton_rtol', type=float, default=1e-3,
                        help='[SnareNet] Newton solver relative tolerance')
    parser.add_argument('--lambda_reg', type=float, default=0,
                        help='[SnareNet] Tikhonov regularization parameter')
    parser.add_argument('--if_project', action='store_true',
                        help='[SnareNet] enable constraint projection')
    parser.add_argument('--no_project', dest='if_project', action='store_false',
                        help='[SnareNet] disable constraint projection')
    parser.set_defaults(if_project=True)
    parser.add_argument('--soft_epochs', type=int, default=0,
                        help='[SnareNet] disable projection for first N epochs (soft training)')

    # Extract arguments from parser
    args = parser.parse_args()
    # set initial seed
    initial_seed = args.seed
    set_random_seed(initial_seed)

    # organize arguments for the experiment into a dictionary for logging purpose
    experiment_args = {
        "model": args.model,
        "initial_seed": args.seed,
        "pde": args.pde,
        "pde_params": args.pde_params,
        "opt": args.opt,
        "opt_params": args.opt_params,
        "num_layers": args.num_layers,
        "num_neurons": args.num_neurons,
        "loss": args.loss,
        "data_loss_weight": args.data_loss_weight,
        "num_data_samples": args.num_data_samples,
        "num_x": args.num_x,
        "num_t": args.num_t,
        "num_res": args.num_res, 
        "epochs": args.epochs,
        "wandb_project": args.wandb_project,
        "device": f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    }
    
    # Add SnareNet-specific parameters if using snarenet model
    if args.model == 'snarenet':
        experiment_args.update({
            "newton_maxiter": args.newton_maxiter,
            "newton_rtol": args.newton_rtol,
            "lambda_reg": args.lambda_reg,
            "if_project": args.if_project,
            "soft_epochs": args.soft_epochs
        })

    # print out arguments
    print("="*60)
    print(f"Model: {experiment_args['model'].upper()}")
    print("Seed set to: {}".format(initial_seed))
    print("Selected PDE type: {}".format(experiment_args["pde"]))
    print("Specified PDE coefficients: {}".format(
        experiment_args["pde_params"]))
    print("Optimizer to use: {}".format(experiment_args["opt"]))
    print("Specified optimizer parameters: {}".format(
        experiment_args["opt_params"]))
    print("Number of layers: {}".format(experiment_args["num_layers"]))
    print("Number of neurons per layer: {}".format(experiment_args["num_neurons"]))
    print("Number of spatial points (x): {}".format(experiment_args["num_x"]))
    print("Number of temporal points (t): {}".format(experiment_args["num_t"]))
    print("Number of random residual points to sample: {}".format(experiment_args["num_res"]))
    print("Number of epochs: {}".format(experiment_args["epochs"]))
    print("Weights and Biases project: {}".format(
        experiment_args["wandb_project"]))
    print("GPU to use: {}".format(experiment_args["device"]))
    
    if args.model == 'snarenet':
        print("\nSnareNet Settings:")
        print(f"  Projection enabled: {experiment_args['if_project']}")
        print(f"  Soft epochs: {experiment_args['soft_epochs']}")
        print(f"  Newton max iterations: {experiment_args['newton_maxiter']}")
        print(f"  Newton rtol: {experiment_args['newton_rtol']}")
        print(f"  Lambda regularization: {experiment_args['lambda_reg']}")
    print("="*60)

    # Generate wandb run name
    wandb_name = set_wandb_run_name(
        model=experiment_args["model"],
        seed=experiment_args["initial_seed"],
        pde_name=experiment_args["pde"],
        pde_params=experiment_args["pde_params"]
    )

    with wandb.init(project=experiment_args["wandb_project"], name=wandb_name, config=experiment_args):
        # Initialize model based on type
        if experiment_args["model"] == "pinn":
            # Standard PINN
            model = PINN(
                in_dim=2, 
                hidden_dim=experiment_args["num_neurons"], 
                out_dim=1,
                num_layer=experiment_args["num_layers"]
            ).to(experiment_args["device"])
            print(f"Standard PINN initialized with {sum(p.numel() for p in model.parameters())} parameters")
            
        elif experiment_args["model"] == "snarenet":
            # SnareNet PINN - need to create constraint first
            x_range, t_range, _, pde_coefs = get_pde(
                experiment_args["pde"],
                experiment_args["pde_params"],
                experiment_args["loss"]
            )
            pde_constraint = get_pde_constraint(
                experiment_args["pde"],
                pde_coefs,
                x_range,
                t_range
            )

            model = SnareNetPINN(
                in_dim=2,
                hidden_dim=experiment_args["num_neurons"],
                out_dim=1,
                num_layer=experiment_args["num_layers"],
                pde_constraint=pde_constraint,
                newton_maxiter=experiment_args["newton_maxiter"],
                rtol=experiment_args["newton_rtol"],
                lambda_reg=experiment_args["lambda_reg"],
                if_project=experiment_args["if_project"]
            ).to(experiment_args["device"])
            print(f"SnareNetPINN initialized:")
            print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
            print(f"  Constraint dimension: {pde_constraint.nconstraints}")
            print(f"  Augmented output dimension: {pde_constraint.ydim}")
        
        else:
            raise ValueError(f"Unknown model type: {experiment_args['model']}")
        
        # Create net modifier function for SnareNet soft training
        net_modifier_fn = None
        if experiment_args["model"] == "snarenet" and experiment_args["soft_epochs"] > 0:
            soft_epochs = experiment_args["soft_epochs"]
            def modify_net(net, epoch):
                if epoch < soft_epochs:
                    net.set_projection(False)
                else:
                    net.set_projection(True)
                return net
            net_modifier_fn = modify_net
            print(f"\nSoft training enabled: Projection will be disabled for first {soft_epochs} epochs")
        
        # Train the model (train function handles both PINN and SnareNetPINN)
        try:
            train(model,
                  proj_name=experiment_args["wandb_project"],
                  pde_name=experiment_args["pde"],
                  pde_params=experiment_args["pde_params"],
                  loss_name=experiment_args["loss"],
                  opt_name=experiment_args["opt"],
                  opt_params_list=experiment_args["opt_params"],
                  n_x=experiment_args["num_x"],
                  n_t=experiment_args["num_t"],
                  n_res=experiment_args["num_res"],
                  num_epochs=experiment_args["epochs"],
                  device=experiment_args["device"],
                  net_modifier_fn=net_modifier_fn,
                  data_loss_weight=experiment_args["data_loss_weight"],
                  num_data_samples=experiment_args["num_data_samples"])
            print("\nTraining completed successfully!")
            
            # Save model checkpoint
            if experiment_args["model"] == "snarenet":
                save_checkpoint(model, experiment_args, wandb_name, pde_coefs, x_range, t_range)
            else:
                save_checkpoint(model, experiment_args, wandb_name)
            
            
        # log error and traceback info to W&B, and exit gracefully
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e

if __name__ == "__main__":
    main()