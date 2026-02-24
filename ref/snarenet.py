"""
SnareNet: Neural network with Newton-based projection for hard constraints
"""

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce
import time

# available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# print(f"Available GPUs: {available_gpus}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SnareNet(nn.Module):
    """Newton-based neural network for hard-constrained optimization"""
    
    def __init__(self, data, cfg):
        super().__init__()
        self._data = data
        self._cfg = cfg
        self._if_project = False
        self._eps = torch.zeros(data.nineq + data.neq, device=DEVICE)  # Epsilon for adaptive relaxation (num_constraints,)
        self._iter_taken = 0
        
        # Newton and CG settings - read from model config
        model_config = cfg.model
        self._newton_maxiter = cfg.model['newton_maxiter']  # Maximum Newton iterations
        self._rtol = cfg.model['rtol']  # Relative tolerance for Newton convergence
        self._if_cg = cfg.model['is_cg']  # Use pseudo-inverse by default
        self._cg_maxiter = cfg.model['cg_maxiter']  # Maximum CG iterations
        self._trust_region = cfg.model['trust_region']
        self._lambda = torch.tensor(cfg.model['lambd'], device=DEVICE)  # Regularization parameter for Newton step

        layer_sizes = [data.encoded_xdim, model_config['hidden_size'], model_config['hidden_size']]
        output_dim = data.ydim

        layers = reduce(operator.add,
            [[nn.Linear(a,b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        # layers = reduce(operator.add,
        #     [[nn.Linear(a,b), nn.ReLU()]
        #         for a,b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        
        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self._net = nn.Sequential(*layers)
    
    def set_projection(self, val=True):
        """Set whether to do projection or not"""
        self._if_project = val
    
    def set_eps(self, eps):
        """Set epsilon for adaptive relaxation (used during training)"""
        self._eps = eps
    
    def get_eps(self):
        """Get current epsilon value"""
        return self._eps

    def get_iter_taken(self):
        """Get number of Newton iterations taken in last projection"""
        return self._iter_taken

    def apply_newton(self, f, x):
        """Project f to satisfy bl<=Af<=bu using Newton's method"""
        bl, bu = self._data.get_lower_upper_bounds(x)
        g = self._data.get_g(x)
        J = self._data.get_jacobian(x)
        
        y_new = f
        y_old = f

        for i in range(self._newton_maxiter):
            g_y = g(y_new)
            J_y = J(y_new)
            f_y = -(nn.ReLU()(bl - self._eps - g_y) - nn.ReLU()(g_y - bu - self._eps))

            if torch.max(torch.abs(f_y)) < self._rtol:
                self._iter_taken = i + 1
                return y_new

            # Compute inverse by CG
            if self._if_cg:
                # Tikhonov regularization
                if self._lambda > 0:
                    def JTJ_reg_bmm(v):
                        """Computes (J_y^T @ J_y + lambda * I) @ v for batch"""
                        # v: size = [batch_size, m, k]; J_y: size = [batch_size, m, n]
                        temp = torch.bmm(J_y, v)    # (batch_size, m, k)
                        result = torch.bmm(J_y.transpose(1, 2), temp)   # (batch_size, n, k)
                        result = result + self._lambda * v
                        return result
                    
                    # Precompute diagonal for preconditioner (only depends on J_y)
                    diag_JTJreg = (J_y ** 2).sum(dim=1) + self._lambda  # shape (batch_size, n)
                    def PJTJreg_bmm(v):
                        """
                        Diagonal/Jacobi preconditioner: 1 / diag(J_y^T J_y + lambda * I) @ v
                        """
                        return v / diag_JTJreg.unsqueeze(-1)
                    
                    # Run CG
                    temp = torch.bmm(J_y.transpose(1, 2), f_y[:,:, None])

                    update = self.cg_batch(
                        A_bmm=JTJ_reg_bmm,
                        B=temp,
                        P_bmm=PJTJreg_bmm,
                        maxiter=self._cg_maxiter,
                    ).squeeze(-1)
                
                # No regularization (and must be undertermined case)
                else:
                    def JJT_bmm(v):
                        """Computes J_y @ J_y^T @ v for batch"""
                        # v: size = [batch_size, m, k]; J_y: size = [batch_size, m, n]
                        temp = torch.bmm(J_y.transpose(1, 2), v)    # (batch_size, n, k)
                        result = torch.bmm(J_y, temp)   # (batch_size, m, k)
                        return result

                    # Precompute diagonal for preconditioner (only depends on J_y)
                    diag_JJT = (J_y ** 2).sum(dim=2) + 1e-8  # shape (batch_size, m)
                    def PJJT_bmm(v):
                        """
                        Diagonal/Jacobi preconditioner: 1 / diag(J_y J_y^T) @ v
                        """
                        result = v / diag_JJT.unsqueeze(-1)
                        return result
                    
                    # Run CG
                    temp = self.cg_batch(
                        A_bmm=JJT_bmm,
                        B=f_y[:,:, None],
                        P_bmm=PJJT_bmm,
                        maxiter=self._cg_maxiter,
                    )
                    update = torch.bmm(J_y.transpose(1, 2), temp).squeeze(-1)
            
            # Use pseudo-inverse
            else:
                # Tikhonov regularization
                if self._lambda > 0:
                    m = J_y.shape[2]
                    I = torch.eye(m, device=DEVICE, dtype=J_y.dtype).unsqueeze(0).repeat(J_y.shape[0], 1, 1)
                    J_y_reg = torch.bmm(J_y.transpose(1, 2), J_y) + self._lambda * I
                    J_yT = J_y.transpose(1, 2)
                    RHS = torch.bmm(J_yT, f_y[:,:, None]).squeeze(-1)
                    update = torch.linalg.solve(J_y_reg, RHS)
                # No regularization
                else:
                    J_y_reg = torch.bmm(J_y, J_y.transpose(1, 2))
                    intermediate = torch.linalg.solve(J_y_reg, f_y)
                    J_yT = J_y.transpose(1, 2)
                    update = torch.bmm(J_yT, intermediate[:,:, None]).squeeze(-1)
            
            y_old = y_new
            if self._trust_region:
                # Clip update to avoid large steps
                update = torch.clamp(update, -1, 1)
            
            y_new = y_old - update
            
            # Early stopping if the update is small
            if torch.max(torch.abs(update)) < self._rtol:
                self._iter_taken = i + 1
                return y_new

        return y_new
    
    def cg_batch(self, A_bmm, B, P_bmm=None, rtol=1e-3, atol=0., maxiter=10, X0=None):
        """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

        This function solves a batch of matrix linear systems of the form

            A_i X_i = B_i,  i=1,...,K,

        where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
        and X_i is the n x m matrix representing the solution for the ith system.

        Args:
            A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
            B: A K x n x m matrix representing the right hand sides.
            P_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
                matrices P and a K x n x m matrix. (default=identity matrix)
            X0: (optional) Initial guess for X, defaults to P_bmm(B). (default=None)
            rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
            atol: (optional) Absolute tolerance for norm of residual. (default=0)
            maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
            verbose: (optional) Whether or not to print status messages. (default=False)
        
        Code adapted from: https://github.com/sbarratt/torch_cg/blob/master/torch_cg/cg_batch.py
        """
        K, n, m = B.shape

        if P_bmm is None:
            P_bmm = lambda x: x
        if X0 is None:
            X0 = P_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        assert B.shape == (K, n, m)
        assert X0.shape == (K, n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)

        X_k = X0
        R_k = B - A_bmm(X_k)
        Z_k = P_bmm(R_k)

        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

        for k in range(1, maxiter + 1):
            Z_k = P_bmm(R_k)

            if k == 1:
                P_k = Z_k
                R_k1 = R_k
                X_k1 = X_k
                Z_k1 = Z_k
            else:
                R_k2 = R_k1
                Z_k2 = Z_k1
                P_k1 = P_k
                R_k1 = R_k
                Z_k1 = Z_k
                X_k1 = X_k
                denominator = (R_k2 * Z_k2).sum(1)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(1) / denominator
                P_k = Z_k1 + beta.unsqueeze(1) * P_k1

            denominator = (P_k * A_bmm(P_k)).sum(1)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(1) / denominator
            X_k = X_k1 + alpha.unsqueeze(1) * P_k
            R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)

            residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

            if (residual_norm <= stopping_matrix).all():
                break

        return X_k

    def forward(self, x):
        encoded_x = self._data.encode_input(x)
        out = self._net(encoded_x)

        if self._if_project:
            out = self.apply_newton(out, x)
        
        return out
