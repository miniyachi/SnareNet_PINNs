## Usage:
# python3 run_oplr.py --seed 123 --pde convection --pde_params beta 1 \
#   --loss mse --opt_params lr 0.001 \
#   --n_x 257 --n_t 101 --n_res 10000 \
#   --n_funcs 100 --funcspace_params length_scale 0.2 \
#   --model_type DeepONetCartesianProd --branch_hidden 64 64 --trunk_hidden 64 64 \
#   --epochs 40000 --wandb_project test_oplr --device 0

import wandb
import argparse
import torch
import deepxde as dde

from src.train_utils import set_random_seed
from src.oplr_utils import get_model_oplr, train_oplr, set_float_precision


def set_wandb_run_name(seed, pde_name, pde_params):
    params_str = "_".join(pde_params) if pde_params else "default"
    return f"oplr_seed{seed}_{pde_name}_{params_str}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',             type=int,   default=1234)
    parser.add_argument('--pde',              type=str,   default='convection')
    parser.add_argument('--pde_params',       nargs='+',  type=str, default=None)
    parser.add_argument('--loss',             type=str,   default='mse')
    parser.add_argument('--opt_params',       nargs='+',  type=str, default=None,
                        help='Adam optimizer params, e.g. lr 0.001')
    parser.add_argument('--n_sensor',        type=int,   default=100,
                        help='Number of branch sensor points')
    parser.add_argument('--n_x',             type=int,   default=257,
                        help='Number of spatial collocation points')
    parser.add_argument('--n_t',             type=int,   default=101,
                        help='Number of temporal collocation points')
    parser.add_argument('--n_res',           type=int,   default=10000,
                        help='Number of interior collocation points to sample for training')
    parser.add_argument('--n_funcs',       type=int,   default=100,
                        help='Number of GRF functions to sample')
    parser.add_argument('--funcspace_params',       nargs='+',  type=str, default=None,
                        help='GRF params, e.g. length_scale 0.2')
    parser.add_argument('--model_type',      type=str,   default='DeepONetCartesianProd',
                        choices=['DeepONetCartesianProd'],
                        help='Neural operator architecture')
    parser.add_argument('--branch_hidden',   nargs='+',  type=int, default=[64, 64],
                        help='Hidden layer sizes for branch network, e.g. 64 64')
    parser.add_argument('--trunk_hidden',    nargs='+',  type=int, default=[64, 64],
                        help='Hidden layer sizes for trunk network, e.g. 64 64')
    parser.add_argument('--activation',      type=str,   default='sin',
                        help='Activation function for model, e.g. sin, relu, tanh etc.')
    parser.add_argument('--epochs',          type=int,   default=4000)
    parser.add_argument('--wandb_project',   type=str,   default='oplr')
    parser.add_argument('--device',          type=str,   default='0')
    parser.add_argument('--float',           type=int,   default=32, choices=[32, 64],
                        help='Floating point precision for training, e.g. 32 or 64')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    set_random_seed(args.seed)
    set_float_precision(args.float)

    experiment_args = vars(args)
    experiment_args['device'] = device

    print("=" * 60)
    print(f"Operator Learning")
    print(f"  Seed:            {args.seed}")
    print(f"  Float precision: float{args.float}")
    print(f"  PDE:             {args.pde}  params={args.pde_params}")
    print(f"  Loss:            {args.loss}")
    print(f"  Optimizer:       Adam  params={args.opt_params}")
    print(f"  Grid:            n_x={args.n_x}, n_t={args.n_t}, n_res={args.n_res}")
    print(f"  GRF:             n_funcs={args.n_funcs}, params={args.funcspace_params}")
    print(f"  Model type:      {args.model_type}")
    print(f"  Branch hidden:   {args.branch_hidden}")
    print(f"  Trunk hidden:    {args.trunk_hidden}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Device:          {device}")
    print("=" * 60)

    model = get_model_oplr(
        model_type=args.model_type,
        n_sensors=args.n_sensor,    # n_sensor determines the branch network input size
        branch_hidden=args.branch_hidden,
        trunk_hidden=args.trunk_hidden,
        activation=args.activation,
    ).to(device)
    print(f"DeepONet initialized with {sum(p.numel() for p in model.parameters())} parameters")

    wandb_name = set_wandb_run_name(args.seed, args.pde, args.pde_params)
    with wandb.init(project=args.wandb_project, name=wandb_name, config=experiment_args):
        train_oplr(
            model=model,
            pde_name=args.pde,
            pde_params=args.pde_params,
            loss_name=args.loss,
            opt_params_list=args.opt_params,
            funcspace_params_list=args.funcspace_params,
            n_funcs=args.n_funcs,
            n_sensor=args.n_sensor,
            n_x=args.n_x,
            n_t=args.n_t,
            n_res=args.n_res,
            num_epochs=args.epochs,
            device=device,
            model_type=args.model_type,
        )


if __name__ == "__main__":
    main()
