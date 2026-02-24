import torch
import torch.nn as nn

'''
Implementation of PINNs. 
'''

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class PINN(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
    super(PINN, self).__init__()

    layers = []
    for i in range(num_layer-1):
      if i == 0:
        layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
        layers.append(SinActivation())
      else:
        layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        layers.append(SinActivation())

    layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

    self.linear = nn.Sequential(*layers)

  def forward(self, x, t):
    src = torch.cat((x,t), dim=-1)
    return self.linear(src)


class SnareNetPINN(PINN):
  """
  PINN with Newton-based projection for hard constraint enforcement.
  Inherits the base network architecture from PINN and adds projection capability.
  Uses PDEConstraint object with paired inputs for constraint enforcement.
  """
  def __init__(self, in_dim, hidden_dim, out_dim, num_layer, pde_constraint,
               newton_maxiter=10, rtol=1e-3, lambda_reg=0.0, if_project=True):
    """
    Args:
      in_dim: Input dimension (e.g., 2 for x and t)
      hidden_dim: Hidden layer dimension
      out_dim: Output dimension
      num_layer: Number of layers
      pde_constraint: PDEConstraint object for PDE-specific constraints
      newton_maxiter: Maximum Newton iterations for projection
      rtol: Relative tolerance for Newton convergence
      lambda_reg: Regularization parameter for Newton step (Tikhonov regularization)
      if_project: Whether projection is enabled
    """
    super(SnareNetPINN, self).__init__(in_dim, hidden_dim, out_dim, num_layer)

    self.pde_constraint = pde_constraint
    self._if_project = if_project
    self._iter_taken = 0
    self._eps = torch.zeros(pde_constraint.nconstraints)  # Epsilon for adaptive relaxation (num_constraints,)
    
    # Newton solver configuration
    self._newton_maxiter = newton_maxiter
    self._rtol = rtol
    self._lambda_val = lambda_reg  # Store as Python float for comparisons

  def set_projection(self, val=True):
    """Enable or disable projection"""
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
  
  def forward_augmented(self, x1, x2, t):
    """
    Compute augmented output including derivatives for PDE constraints.
    
    The augmented output depends on the PDE type:
    - Convection: [u1, u2, ut1, ut2, ux1, ux2]
    - Reaction: [u1, u2, ut1, ut2]
    - Wave: [u1, u2, ut1, ut2, utt1, utt2, ux1, ux2, uxx1, uxx2]
    
    Args:
      x1, x2: Paired spatial coordinates (batch_size, 1)
      t: Temporal coordinate (batch_size, 1)
    
    Returns:
      y: Augmented output tensor (batch_size, n_outputs)
    """
    
    # Ensure gradients are enabled
    x1 = x1.requires_grad_(True)
    x2 = x2.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Base predictions
    u1 = super().forward(x1, t)  # u(x1, t)
    u2 = super().forward(x2, t)  # u(x2, t)
    
    # Compute first-order derivatives for both points
    ut1 = torch.autograd.grad(u1, t, grad_outputs=torch.ones_like(u1),
                              create_graph=True, retain_graph=True)[0]
    ux1 = torch.autograd.grad(u1, x1, grad_outputs=torch.ones_like(u1),
                              create_graph=True, retain_graph=True)[0]
    
    ut2 = torch.autograd.grad(u2, t, grad_outputs=torch.ones_like(u2),
                              create_graph=True, retain_graph=True)[0]
    ux2 = torch.autograd.grad(u2, x2, grad_outputs=torch.ones_like(u2),
                              create_graph=True, retain_graph=True)[0]
    
    # Build augmented output based on PDE type
    if self.pde_constraint.pde_name == "convection":
      # Augmented: [u1, u2, ut1, ut2, ux1, ux2]
      y = torch.cat([u1, u2, ut1, ut2, ux1, ux2], dim=-1)
      
    # elif self.pde_constraint.pde_name == "reaction":
    #   # Augmented: [u1, u2, ut1, ut2]
    #   y = torch.cat([u1, u2, ut1, ut2], dim=-1)
      
    # elif self.pde_constraint.pde_name == "wave":
    #   # Need second-order derivatives
    #   utt1 = torch.autograd.grad(ut1, t, grad_outputs=torch.ones_like(ut1),
    #                             create_graph=True, retain_graph=True)[0]
    #   uxx1 = torch.autograd.grad(ux1, x1, grad_outputs=torch.ones_like(ux1),
    #                             create_graph=True, retain_graph=True)[0]
      
    #   utt2 = torch.autograd.grad(ut2, t, grad_outputs=torch.ones_like(ut2),
    #                             create_graph=True, retain_graph=True)[0]
    #   uxx2 = torch.autograd.grad(ux2, x2, grad_outputs=torch.ones_like(ux2),
    #                             create_graph=True, retain_graph=True)[0]
      
    #   # Augmented: [u1, u2, ut1, ut2, utt1, utt2, ux1, ux2, uxx1, uxx2]
    #   y = torch.cat([u1, u2, ut1, ut2, utt1, utt2, ux1, ux2, uxx1, uxx2], dim=-1)
      
    else:
      raise ValueError(f"Unknown PDE type: {self.pde_constraint.pde_name}")
    
    return y

  def repair(self, out, x1, x2, t):
    """
    Project network output to satisfy constraints using Newton's method.
    
    Args:
      out: Augmented network output to be projected (batch_size, n_outputs)
      x1: First spatial input (batch_size, 1)
      x2: Second spatial input (batch_size, 1)
      t: Temporal input (batch_size, 1)
    
    Returns:
      Projected output satisfying constraints
    """
    # Get constraint functions from PDE constraint
    bl, bu = self.pde_constraint.get_lower_upper_bounds(x1, x2, t)
    g = self.pde_constraint.get_g(x1, x2, t)
    J = self.pde_constraint.get_jacobian(x1, x2, t)
    
    y = out

    for i in range(self._newton_maxiter):
      g_y = g(y)
      J_y = J(y)
      
      # Constraint violation: r_y represents how much (relaxed) constraints are violated
      r_y = -(nn.ReLU()(bl - self._eps - g_y) - nn.ReLU()(g_y - bu - self._eps))

      # Check convergence
      if torch.amax(torch.abs(r_y)) < self._rtol:
        self._iter_taken = i
        return y

      # Compute Newton update using pseudo-inverse
      dy = self._compute_update_pinv(J_y, r_y)
      
      y = y - dy
      
      # Early stopping if update is small
      if torch.amax(torch.abs(dy)) < self._rtol:
        self._iter_taken = i + 1
        return y

    self._iter_taken = self._newton_maxiter
    return y

  def _compute_update_pinv(self, J_y, r_y):
    """
    Compute Newton update using pseudo-inverse.
    
    Args:
      J_y: Jacobian matrix of constraints with respect to output, shape (batch_size, m, n)
           where m is the number of constraints and n is the output dimension
      r_y: Constraint residual vector, shape (batch_size, m)
           where m is the number of constraints
    
    Returns:
      dy: Newton update step, shape (batch_size, n) where n is the output dimension
    """
    # Tikhonov regularization
    if self._lambda_val > 0:
      # Compute transpose once and reuse
      J_yT = J_y.transpose(1, 2)
      JTJ_reg = torch.bmm(J_yT, J_y)
      
      # Add regularization to diagonal (more efficient than creating full identity matrix)
      JTJ_reg.diagonal(dim1=1, dim2=2).add_(self._lambda_val)
      
      # Solve for update
      rhs = torch.bmm(J_yT, r_y.unsqueeze(-1))
      dy = torch.linalg.solve(JTJ_reg, rhs).squeeze(-1)
    
    # No regularization - use Moore-Penrose pseudo-inverse
    else:
      J_yT = J_y.transpose(1, 2)
      JJT = torch.bmm(J_y, J_yT)
      
      intermediate = torch.linalg.solve(JJT, r_y.unsqueeze(-1))
      dy = torch.bmm(J_yT, intermediate).squeeze(-1)
    
    return dy

  def forward(self, x1, x2, t):
    """
    Forward pass through PINN with paired inputs and optional projection.
    
    Args:
      x1: First spatial coordinate (batch_size, 1)
      x2: Second spatial coordinate (batch_size, 1)
      t: Temporal coordinate (batch_size, 1)
    
    Returns:
      Tuple (u1, u2) with shapes (batch_size, 1) each, representing
      u(x1,t) and u(x2,t) after constraint projection
    """
    # Get augmented output
    y = self.forward_augmented(x1, x2, t)
    
    # Apply projection if enabled
    if self._if_project:
      y = self.repair(y, x1, x2, t)
    
    # Extract u1, u2 from augmented output (first two components)
    u1 = y[:, 0:1]
    u2 = y[:, 1:2]
    
    return u1, u2
