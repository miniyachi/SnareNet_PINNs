import torch
import torch.nn as nn
import numpy as np

"""
PDEConstraint class for defining constraints on PINN solutions.
Handles constraint functions, Jacobians, and bounds for different PDEs.
"""

class PDEConstraint:
    """
    Defines constraints for a specific PDE problem.
    Handles constraint function g, Jacobian, and bounds.
    
    The constraint is formulated as: bl <= g(y) <= bu
    where y is the augmented output [u1, u2, derivatives...]
    """
    
    def __init__(self, pde_name, pde_coefs, x_range, t_range):
        """
        Args:
            pde_name: Name of the PDE (e.g., 'convection', 'reaction', 'wave')
            pde_coefs: Dictionary of PDE coefficients
            x_range: [x_min, x_max] spatial domain
            t_range: [t_min, t_max] temporal domain
        """
        self.pde_name = pde_name
        self.pde_coefs = pde_coefs
        self.x_range = x_range
        self.t_range = t_range
        self.nconstraints = self._get_num_constraints()
        self.ydim = self._get_ydim()  # Dimension of augmented output y
        
    def _get_num_constraints(self):
        """
        Determine number of equality constraints for the PDE.
        This dynamically counts constraints by checking which ones are active.
        """
        if self.pde_name == "convection":
            # Count by checking the constraint configuration
            return self._count_active_constraints_convection()
        # elif self.pde_name == "reaction":
        #     return self._count_active_constraints_reaction()
        # elif self.pde_name == "wave":
        #     return self._count_active_constraints_wave()
        else:
            raise ValueError(f"Unknown PDE: {self.pde_name}")
    
    def _get_ydim(self):
        """Determine dimension of augmented output y for the PDE"""
        if self.pde_name == "convection":
            # y = [u1, u2, ut1, ut2, ux1, ux2]
            return 6
        # elif self.pde_name == "reaction":
        #     # y = [u1, u2, ut1, ut2]
        #     return 4
        # elif self.pde_name == "wave":
        #     # y = [u1, u2, ut1, ut2, utt1, utt2, ux1, ux2, uxx1, uxx2]
        #     return 10
        else:
            raise ValueError(f"Unknown PDE: {self.pde_name}")
    
    def get_g(self, x1, x2, t):
        """
        Returns constraint function g that takes augmented output y.
        
        Args:
            x1, x2: Spatial coordinates (batch_size, 1)
            t: Temporal coordinate (batch_size, 1)
        
        Returns:
            Function handle g(y) where y shape: (batch_size, n_outputs)
        """
        if self.pde_name == "convection":
            return self._get_g_convection(x1, x2, t)
        # elif self.pde_name == "reaction":
        #     return self._get_g_reaction(x1, x2, t)
        # elif self.pde_name == "wave":
        #     return self._get_g_wave(x1, x2, t)
        else:
            raise ValueError(f"Unknown PDE: {self.pde_name}")
    
    def get_jacobian(self, x1, x2, t):
        """
        Returns Jacobian dg/dy.
        
        Args:
            x1, x2: Spatial coordinates (batch_size, 1)
            t: Temporal coordinate (batch_size, 1)
        
        Returns:
            Function handle J(y) that returns Jacobian matrix
        """
        if self.pde_name == "convection":
            return self._get_jacobian_convection(x1, x2, t)
        # elif self.pde_name == "reaction":
        #     return self._get_jacobian_reaction(x1, x2, t)
        # elif self.pde_name == "wave":
        #     return self._get_jacobian_wave(x1, x2, t)
        else:
            raise ValueError(f"Unknown PDE: {self.pde_name}")
    
    def get_lower_upper_bounds(self, x1, x2, t):
        """
        Returns (bl, bu) for constraints bl <= g(y) <= bu.
        For equality constraints: bl = bu.
        
        Args:
            x1, x2: Spatial coordinates (batch_size, 1)
            t: Temporal coordinate (batch_size, 1)
        
        Returns:
            (bl, bu) tensors of shape (batch_size, n_constraints)
        """
        if self.pde_name == "convection":
            return self._get_bounds_convection(x1, x2, t)
        # elif self.pde_name == "reaction":
        #     return self._get_bounds_reaction(x1, x2, t)
        # elif self.pde_name == "wave":
        #     return self._get_bounds_wave(x1, x2, t)
        else:
            raise ValueError(f"Unknown PDE: {self.pde_name}")
    
    # ========== Convection PDE Methods ==========

    def _count_active_constraints_convection(self):
        """Count the number of active constraints by checking the configuration."""
        count = 0
        
        # # PDE at point 1: ut1 + beta*ux1 = 0
        # count += 1
        
        # # PDE at point 2: ut2 + beta*ux2 = 0
        # count += 1
        
        # Periodic BC: u1 - u2 = 0
        count += 1
        
        # Initial condition at x1: u1 - sin(x1) = 0
        count += 1
        
        # Initial condition at x2: u2 - sin(x2) = 0
        count += 1
        
        # # Consistency ut1 == ut2 when x1 == x2
        # count += 1
        
        # # Consistency ux1 == ux2 when x1 == x2
        # count += 1
        
        return count
    
    def _get_g_convection(self, x1, x2, t):
        """
        Convection PDE: u_t + beta * u_x = 0
        IC: u(x, 0) = sin(x)
        BC: u(0, t) = u(2*pi, t) (periodic)
        
        Augmented output y = [u1, u2, ut1, ut2, ux1, ux2]
        where subscript 1/2 refers to evaluation at x1/x2
        """
        beta = self.pde_coefs["beta"]
        device = x1.device
        
        def g(y):
            # y: (batch_size, 6) = [u1, u2, ut1, ut2, ux1, ux2]
            batch_size = y.shape[0]
            constraints = []
            
            # # PDE at point 1: ut1 + beta*ux1 = 0
            # constraints.append(y[:, 2] + beta * y[:, 4])
            
            # # PDE at point 2: ut2 + beta*ux2 = 0
            # constraints.append(y[:, 3] + beta * y[:, 5])
            
            # Periodic BC / Consistency: u1 - u2 = 0
            # - Periodic BC active when x1=0, x2=2π
            # - Consistency active when x1=x2 in interior
            constraints.append(y[:, 0] - y[:, 1])
            
            # Initial condition at x1: u1 - sin(x1) = 0
            constraints.append(y[:, 0] - torch.sin(x1.squeeze(-1)))
            
            # Initial condition at x2: u2 - sin(x2) = 0
            constraints.append(y[:, 1] - torch.sin(x2.squeeze(-1)))
            
            # # Consistency ut1 == ut2 when x1 == x2
            # constraints.append(y[:, 2] - y[:, 3])
            
            # # Consistency ux1 == ux2 when x1 == x2
            # constraints.append(y[:, 4] - y[:, 5])
            
            # Stack all constraints into a tensor
            g_val = torch.stack(constraints, dim=1)
            return g_val
        
        return g
    
    def _get_jacobian_convection(self, x1, x2, t):
        """
        Jacobian of convection constraints.
        Since constraints are linear in y, Jacobian is constant.
        """
        beta = self.pde_coefs["beta"]
        device = x1.device
        batch_size = x1.shape[0]
        
        def J(y):
            # Jacobian shape: (batch_size, n_constraints, n_outputs)
            jacobians = []
            
            # # Row for: ut1 + beta*ux1
            # j_row = torch.zeros(batch_size, self.ydim, device=device)
            # j_row[:, 2] = 1.0
            # j_row[:, 4] = beta
            # jacobians.append(j_row)
            
            # # Row for: ut2 + beta*ux2
            # j_row = torch.zeros(batch_size, self.ydim, device=device)
            # j_row[:, 3] = 1.0
            # j_row[:, 5] = beta
            # jacobians.append(j_row)
            
            # Row for: u1 - u2
            j_row = torch.zeros(batch_size, self.ydim, device=device)
            j_row[:, 0] = 1.0
            j_row[:, 1] = -1.0
            jacobians.append(j_row)
            
            # Row for: u1 - sin(x1)
            j_row = torch.zeros(batch_size, self.ydim, device=device)
            j_row[:, 0] = 1.0
            jacobians.append(j_row)
            
            # Row for: u2 - sin(x2)
            j_row = torch.zeros(batch_size, self.ydim, device=device)
            j_row[:, 1] = 1.0
            jacobians.append(j_row)
            
            # # Row for: ut1 - ut2
            # j_row = torch.zeros(batch_size, self.ydim, device=device)
            # j_row[:, 2] = 1.0
            # j_row[:, 3] = -1.0
            # jacobians.append(j_row)
            
            # # Row for: ux1 - ux2
            # j_row = torch.zeros(batch_size, self.ydim, device=device)
            # j_row[:, 4] = 1.0
            # j_row[:, 5] = -1.0
            # jacobians.append(j_row)
            
            # Stack all jacobian rows
            J_matrix = torch.stack(jacobians, dim=1)
            return J_matrix
        
        return J
    
    def _get_bounds_convection(self, x1, x2, t):
        """
        Bounds for convection constraints.
        We use a masking approach: inactive constraints have very loose bounds.
        """
        device = x1.device
        batch_size = x1.shape[0]
        
        # Determine which constraints are active based on data type
        eps_spatial = 1e-6
        eps_temporal = 1e-6
        
        # Check for boundary points (x1 ≈ 0, x2 ≈ 2π)
        is_boundary = (torch.abs(x1 - self.x_range[0]) < eps_spatial) & \
                     (torch.abs(x2 - self.x_range[1]) < eps_spatial)
        
        # Check for initial points (t ≈ 0)
        is_initial = (torch.abs(t - self.t_range[0]) < eps_temporal)
        
        # Check for interior points (x1 == x2, both in interior)
        is_x_interior = (torch.abs(x1 - x2) < eps_spatial) & \
                     (torch.abs(x1 - self.x_range[0]) > eps_spatial) & \
                     (torch.abs(x1 - self.x_range[1]) > eps_spatial)
        
        # Check for PDE interior points (not at initial time AND at spatial interior)
        is_pde_interior = (~is_initial) & is_x_interior
        
        # Build bounds as lists (same order as constraints in g())
        bl_list = []
        bu_list = []
        
        # # PDE at point 1: only active at interior points
        # bl_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            -torch.inf * torch.ones_like(x1.squeeze(-1))))
        # bu_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # # PDE at point 2: only active at interior points
        # bl_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            -torch.inf * torch.ones_like(x1.squeeze(-1))))
        # bu_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # BC: only active at boundary
        bl_list.append(torch.where(is_boundary.squeeze(-1), 
                                   torch.zeros_like(x1.squeeze(-1)), 
                                   -torch.inf * torch.ones_like(x1.squeeze(-1))))
        bu_list.append(torch.where(is_boundary.squeeze(-1), 
                                   torch.zeros_like(x1.squeeze(-1)), 
                                   torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # IC at x1: only active at initial time
        bl_list.append(torch.where(is_initial.squeeze(-1), 
                                   torch.zeros_like(x1.squeeze(-1)), 
                                   -torch.inf * torch.ones_like(x1.squeeze(-1))))
        bu_list.append(torch.where(is_initial.squeeze(-1), 
                                   torch.zeros_like(x1.squeeze(-1)), 
                                   torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # IC at x2: only active at initial time
        bl_list.append(torch.where(is_initial.squeeze(-1), 
                                   torch.zeros_like(x1.squeeze(-1)), 
                                   -torch.inf * torch.ones_like(x1.squeeze(-1))))
        bu_list.append(torch.where(is_initial.squeeze(-1), 
                                   torch.zeros_like(x1.squeeze(-1)), 
                                   torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # # ut1 == ut2: only active when x1 == x2
        # bl_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            -torch.inf * torch.ones_like(x1.squeeze(-1))))
        # bu_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # # ux1 == ux2: only active when x1 == x2
        # bl_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            -torch.inf * torch.ones_like(x1.squeeze(-1))))
        # bu_list.append(torch.where(is_pde_interior.squeeze(-1), 
        #                            torch.zeros_like(x1.squeeze(-1)), 
        #                            torch.inf * torch.ones_like(x1.squeeze(-1))))
        
        # Stack all bounds
        bl = torch.stack(bl_list, dim=1)
        bu = torch.stack(bu_list, dim=1)
        
        return bl, bu
    
    # # ========== Reaction PDE Methods ==========
    # # TODO: Implement these based on similar pattern to convection
    
    # def _get_g_reaction(self, x1, x2, t):
    #     """
    #     Reaction PDE: u_t - rho * u * (1 - u) = 0
    #     BC: u(0, t) = u(2*pi, t) (periodic)
    #     IC: u(x, 0) = exp(-0.5 * ((x - pi) / (pi/4))^2)
        
    #     NOTE: This constraint is NONLINEAR in y, so Jacobian depends on y.
    #     Augmented output y = [u1, u2, ut1, ut2]
    #     """
    #     rho = self.pde_coefs["rho"]
    #     device = x1.device
        
    #     def g(y):
    #         batch_size = y.shape[0]
    #         n_constraints = 6
    #         g_val = torch.zeros(batch_size, n_constraints, device=device)
            
    #         # Constraint 0 & 1: PDE (nonlinear!)
    #         g_val[:, 0] = y[:, 2] - rho * y[:, 0] * (1 - y[:, 0])  # ut1 - rho*u1*(1-u1)
    #         g_val[:, 1] = y[:, 3] - rho * y[:, 1] * (1 - y[:, 1])  # ut2 - rho*u2*(1-u2)
            
    #         # Constraint 2: Periodic BC
    #         g_val[:, 2] = y[:, 0] - y[:, 1]
            
    #         # Constraints 3 & 4: IC
    #         ic_val1 = torch.exp(-0.5 * torch.square((x1.squeeze(-1) - np.pi) / (np.pi / 4)))
    #         ic_val2 = torch.exp(-0.5 * torch.square((x2.squeeze(-1) - np.pi) / (np.pi / 4)))
    #         g_val[:, 3] = y[:, 0] - ic_val1
    #         g_val[:, 4] = y[:, 1] - ic_val2
            
    #         # Constraint 5: u1 == u2 when x1 == x2
    #         g_val[:, 5] = y[:, 0] - y[:, 1]
            
    #         return g_val
        
    #     return g
    
    # def _get_jacobian_reaction(self, x1, x2, t):
    #     """
    #     Jacobian for reaction PDE (NONLINEAR - depends on y).
    #     """
    #     rho = self.pde_coefs["rho"]
    #     device = x1.device
    #     batch_size = x1.shape[0]
        
    #     def J(y):
    #         # Jacobian shape: (batch_size, 6, 4)
    #         J_matrix = torch.zeros(batch_size, 6, 4, device=device)
            
    #         # Row 0: d(ut1 - rho*u1*(1-u1))/dy
    #         # = [rho*(2*u1 - 1), 0, 1, 0]
    #         J_matrix[:, 0, 0] = rho * (2 * y[:, 0].squeeze(-1) - 1)
    #         J_matrix[:, 0, 2] = 1.0
            
    #         # Row 1: d(ut2 - rho*u2*(1-u2))/dy
    #         J_matrix[:, 1, 1] = rho * (2 * y[:, 1].squeeze(-1) - 1)
    #         J_matrix[:, 1, 3] = 1.0
            
    #         # Row 2: d(u1 - u2)/dy = [1, -1, 0, 0]
    #         J_matrix[:, 2, 0] = 1.0
    #         J_matrix[:, 2, 1] = -1.0
            
    #         # Row 3: d(u1 - ic1)/dy = [1, 0, 0, 0]
    #         J_matrix[:, 3, 0] = 1.0
            
    #         # Row 4: d(u2 - ic2)/dy = [0, 1, 0, 0]
    #         J_matrix[:, 4, 1] = 1.0
            
    #         # Row 5: d(u1 - u2)/dy = [1, -1, 0, 0]
    #         J_matrix[:, 5, 0] = 1.0
    #         J_matrix[:, 5, 1] = -1.0
            
    #         return J_matrix
        
    #     return J
    
    # def _get_bounds_reaction(self, x1, x2, t):
    #     """Bounds for reaction constraints (similar to convection)"""
    #     device = x1.device
    #     batch_size = x1.shape[0]
    #     n_constraints = 6
        
    #     bl = torch.zeros(batch_size, n_constraints, device=device)
    #     bu = torch.zeros(batch_size, n_constraints, device=device)
        
    #     eps_spatial = 1e-5
    #     eps_temporal = 1e-5
    #     large_val = 1e10
        
    #     is_boundary = (torch.abs(x1 - self.x_range[0]) < eps_spatial) & \
    #                  (torch.abs(x2 - self.x_range[1]) < eps_spatial)
    #     is_initial = (torch.abs(t - self.t_range[0]) < eps_temporal)
    #     is_interior = (torch.abs(x1 - x2) < eps_spatial) & \
    #                  (torch.abs(x1 - self.x_range[0]) > eps_spatial) & \
    #                  (torch.abs(x1 - self.x_range[1]) > eps_spatial)
        
    #     bl[:, 2] = torch.where(is_boundary.squeeze(-1), bl[:, 2], -large_val)
    #     bu[:, 2] = torch.where(is_boundary.squeeze(-1), bu[:, 2], large_val)
    #     bl[:, 3] = torch.where(is_initial.squeeze(-1), bl[:, 3], -large_val)
    #     bu[:, 3] = torch.where(is_initial.squeeze(-1), bu[:, 3], large_val)
    #     bl[:, 4] = torch.where(is_initial.squeeze(-1), bl[:, 4], -large_val)
    #     bu[:, 4] = torch.where(is_initial.squeeze(-1), bu[:, 4], large_val)
    #     bl[:, 5] = torch.where(is_interior.squeeze(-1), bl[:, 5], -large_val)
    #     bu[:, 5] = torch.where(is_interior.squeeze(-1), bu[:, 5], large_val)
        
    #     return bl, bu
    
    # # ========== Wave PDE Methods ==========
    
    # def _get_g_wave(self, x1, x2, t):
    #     """
    #     Wave PDE: u_tt - 4 * u_xx = 0
    #     BC: u(0, t) = 0, u(1, t) = 0 (Dirichlet)
    #     IC: u(x, 0) = sin(pi*x) + 0.5*sin(beta*pi*x), u_t(x, 0) = 0
        
    #     Augmented output y = [u1, u2, ut1, ut2, utt1, utt2, ux1, ux2, uxx1, uxx2]
    #     This is 10-dimensional!
    #     """
    #     beta = self.pde_coefs["beta"]
    #     device = x1.device
        
    #     def g(y):
    #         batch_size = y.shape[0]
    #         n_constraints = 8
    #         g_val = torch.zeros(batch_size, n_constraints, device=device)
            
    #         # Constraint 0 & 1: PDE (utt - 4*uxx = 0)
    #         g_val[:, 0] = y[:, 4] - 4.0 * y[:, 8]  # utt1 - 4*uxx1
    #         g_val[:, 1] = y[:, 5] - 4.0 * y[:, 9]  # utt2 - 4*uxx2
            
    #         # Constraint 2 & 3: BC (u = 0 at boundaries)
    #         g_val[:, 2] = y[:, 0]  # u1 = 0 at x1=0
    #         g_val[:, 3] = y[:, 1]  # u2 = 0 at x2=1
            
    #         # Constraint 4 & 5: IC position
    #         ic_val1 = torch.sin(np.pi * x1.squeeze(-1)) + 0.5 * torch.sin(beta * np.pi * x1.squeeze(-1))
    #         ic_val2 = torch.sin(np.pi * x2.squeeze(-1)) + 0.5 * torch.sin(beta * np.pi * x2.squeeze(-1))
    #         g_val[:, 4] = y[:, 0] - ic_val1
    #         g_val[:, 5] = y[:, 1] - ic_val2
            
    #         # Constraint 6 & 7: IC velocity (ut = 0)
    #         g_val[:, 6] = y[:, 2]  # ut1 = 0
    #         g_val[:, 7] = y[:, 3]  # ut2 = 0
            
    #         return g_val
        
    #     return g
    
    # def _get_jacobian_wave(self, x1, x2, t):
    #     """Jacobian for wave PDE (linear constraints)"""
    #     device = x1.device
    #     batch_size = x1.shape[0]
        
    #     def J(y):
    #         # Jacobian shape: (batch_size, 8, 10)
    #         J_matrix = torch.zeros(batch_size, 8, 10, device=device)
            
    #         # Row 0: d(utt1 - 4*uxx1)/dy = [0, 0, 0, 0, 1, 0, 0, 0, -4, 0]
    #         J_matrix[:, 0, 4] = 1.0
    #         J_matrix[:, 0, 8] = -4.0
            
    #         # Row 1: d(utt2 - 4*uxx2)/dy = [0, 0, 0, 0, 0, 1, 0, 0, 0, -4]
    #         J_matrix[:, 1, 5] = 1.0
    #         J_matrix[:, 1, 9] = -4.0
            
    #         # Row 2: d(u1)/dy = [1, 0, ...]
    #         J_matrix[:, 2, 0] = 1.0
            
    #         # Row 3: d(u2)/dy = [0, 1, ...]
    #         J_matrix[:, 3, 1] = 1.0
            
    #         # Row 4: d(u1 - ic1)/dy = [1, 0, ...]
    #         J_matrix[:, 4, 0] = 1.0
            
    #         # Row 5: d(u2 - ic2)/dy = [0, 1, ...]
    #         J_matrix[:, 5, 1] = 1.0
            
    #         # Row 6: d(ut1)/dy = [0, 0, 1, 0, ...]
    #         J_matrix[:, 6, 2] = 1.0
            
    #         # Row 7: d(ut2)/dy = [0, 0, 0, 1, ...]
    #         J_matrix[:, 7, 3] = 1.0
            
    #         return J_matrix
        
    #     return J
    
    # def _get_bounds_wave(self, x1, x2, t):
    #     """Bounds for wave constraints"""
    #     device = x1.device
    #     batch_size = x1.shape[0]
    #     n_constraints = 8
        
    #     bl = torch.zeros(batch_size, n_constraints, device=device)
    #     bu = torch.zeros(batch_size, n_constraints, device=device)
        
    #     eps_spatial = 1e-5
    #     eps_temporal = 1e-5
    #     large_val = 1e10
        
    #     # BC: x1 = 0, x2 = 1
    #     is_boundary_lower = (torch.abs(x1 - self.x_range[0]) < eps_spatial)
    #     is_boundary_upper = (torch.abs(x2 - self.x_range[1]) < eps_spatial)
    #     is_initial = (torch.abs(t - self.t_range[0]) < eps_temporal)
        
    #     # Constraints 0 & 1 (PDE): always active
        
    #     # Constraint 2 (u1=0 at x1=0)
    #     bl[:, 2] = torch.where(is_boundary_lower.squeeze(-1), bl[:, 2], -large_val)
    #     bu[:, 2] = torch.where(is_boundary_lower.squeeze(-1), bu[:, 2], large_val)
        
    #     # Constraint 3 (u2=0 at x2=1)
    #     bl[:, 3] = torch.where(is_boundary_upper.squeeze(-1), bl[:, 3], -large_val)
    #     bu[:, 3] = torch.where(is_boundary_upper.squeeze(-1), bu[:, 3], large_val)
        
    #     # Constraints 4-7 (IC): active at t=0
    #     for i in range(4, 8):
    #         bl[:, i] = torch.where(is_initial.squeeze(-1), bl[:, i], -large_val)
    #         bu[:, i] = torch.where(is_initial.squeeze(-1), bu[:, i], large_val)
        
    #     return bl, bu
