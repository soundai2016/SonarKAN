"""
Models: BSplineLayer, SonarKAN (2D), and SmallMLP baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .bspline import make_open_uniform_knots, bspline_basis_matrix_np, bspline_basis_matrix_torch
from .surrogate import thorp_absorption_db_per_km


@dataclass
class BSplineLayerConfig:
    n_basis: int = 23
    degree: int = 3
    xmin: float = 0.0
    xmax: float = 1.0
    # Ridge parameter for physics-init least squares
    ridge_lambda: float = 1e-6


class BSplineLayer(nn.Module):
    """
    A learnable univariate-function layer where each (input -> output) connection is a spline.

    For an input vector x in R^{n_in}, the output is:
        y_o = b_o + sum_i w_{o,i} x_i + sum_i sum_m c_{i,o,m} B_m(x_i)

    The B-spline basis B_m has compact support induced by a clamped knot vector.
    """

    def __init__(self, num_inputs: int, num_outputs: int, cfg: BSplineLayerConfig):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.cfg = cfg

        # Knot vector (shared across inputs/outputs for simplicity)
        knots_np = make_open_uniform_knots(cfg.n_basis, cfg.degree, cfg.xmin, cfg.xmax)
        self.register_buffer("knots", torch.tensor(knots_np, dtype=torch.float32))

        # Spline coefficients: (n_in, n_out, n_basis)
        self.coefficients = nn.Parameter(torch.randn(num_inputs, num_outputs, cfg.n_basis) * 0.05)

        # Optional linear residual term for stability
        self.base_linear = nn.Linear(num_inputs, num_outputs, bias=True)

        # Initialize linear term to small values
        nn.init.zeros_(self.base_linear.weight)
        nn.init.zeros_(self.base_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in)
        if x.dim() != 2 or x.size(1) != self.num_inputs:
            raise ValueError(f"Expected x shape (B,{self.num_inputs}); got {tuple(x.shape)}")

        # basis per input dimension: (B, n_in, n_basis)
        basis_list = []
        for i in range(self.num_inputs):
            Bi = bspline_basis_matrix_torch(x[:, i], self.knots, self.cfg.degree)  # (B, n_basis)
            basis_list.append(Bi.unsqueeze(1))
        basis = torch.cat(basis_list, dim=1)  # (B, n_in, n_basis)

        # Sum over input dimension i and basis index n -> output (batch, out).
        spline_out = torch.einsum("bin,ion->bo", basis, self.coefficients)
        return self.base_linear(x) + spline_out

    @torch.no_grad()
    def set_parameters_from_least_squares(
        self,
        x_grid: np.ndarray,
        y_target: np.ndarray,
        ridge_lambda: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Fit (linear + spline) parameters to a 1D target function using ridge least squares.

        This is primarily used for physics-informed initialization.

        Only supports num_inputs=num_outputs=1 (used by φ_r and φ_f branches).
        """
        if self.num_inputs != 1 or self.num_outputs != 1:
            raise NotImplementedError("LS init currently implemented for 1->1 layers only.")
        lam = self.cfg.ridge_lambda if ridge_lambda is None else ridge_lambda

        x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
        y_target = np.asarray(y_target, dtype=np.float64).reshape(-1)
        if x_grid.shape[0] != y_target.shape[0]:
            raise ValueError("x_grid and y_target must have same length.")

        B = bspline_basis_matrix_np(x_grid, self.knots.cpu().numpy(), self.cfg.degree)  # (N, n_basis)

        # Design matrix: [x, 1, B]
        A = np.concatenate([x_grid.reshape(-1, 1), np.ones((x_grid.size, 1)), B], axis=1)  # (N, 2+n_basis)

        # Ridge solve: (A^T A + lam I) theta = A^T y
        AtA = A.T @ A
        reg = lam * np.eye(AtA.shape[0])
        theta = np.linalg.solve(AtA + reg, A.T @ y_target)  # (2+n_basis,)

        w = float(theta[0])
        b = float(theta[1])
        c = theta[2:].astype(np.float32)

        self.base_linear.weight[:] = torch.tensor([[w]], dtype=self.base_linear.weight.dtype, device=self.base_linear.weight.device)
        self.base_linear.bias[:] = torch.tensor([b], dtype=self.base_linear.bias.dtype, device=self.base_linear.bias.device)
        self.coefficients[:] = torch.tensor(c.reshape(1, 1, -1), dtype=self.coefficients.dtype, device=self.coefficients.device)

        # diagnostics
        y_fit = A @ theta
        rmse = float(np.sqrt(np.mean((y_fit - y_target) ** 2)))
        return {"rmse": rmse, "ridge_lambda": float(lam)}


@dataclass
class SonarKANConfig:
    """Configuration for the 2D SonarKAN used in Fig. 2.

    Notes
    -----
    - ``spline`` uses ``default_factory`` because it is a dataclass instance (mutable).
    - ``gauge_fix`` resolves the additive model's constant-shift non-identifiability by
      enforcing zero-mean components on a reference grid (and compensating in ``bias``).
    """

    # B-spline edge parameterization shared by branches
    spline: BSplineLayerConfig = field(default_factory=BSplineLayerConfig)

    # Physics-informed initialization
    physics_init_grid_n: int = 256
    fc_hz: float = 3000.0
    use_absorption: bool = True

    # Additive-gauge fixing (identifiability)
    gauge_fix_each_epoch: bool = True
    gauge_fix_grid_n: int = 200

    # Sonar-like constant (used as a convenient offset in the surrogate)
    SL_db: float = 180.0


class SonarKAN(nn.Module):
    """
    2D SonarKAN specialized for inputs x=[r_norm, f_norm].

    Forward:
        y_hat = SL + mean_tl + φ_r(r_norm) + φ_f(f_norm) + bias
    where mean_tl is used to fix the gauge (zero-mean components).
    """

    def __init__(self, r_min_m: float, r_max_m: float, cfg: SonarKANConfig):
        super().__init__()
        self.r_min = float(r_min_m)
        self.r_max = float(r_max_m)
        self.cfg = cfg

        self.phi_r = BSplineLayer(1, 1, cfg.spline)
        self.phi_f = BSplineLayer(1, 1, cfg.spline)
        self.bias = nn.Parameter(torch.zeros(1))

        # Stored after physics_init
        self.register_buffer("mean_tl", torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def physics_init(self) -> Dict[str, float]:
        """
        Initialize φ_r to match spherical spreading + absorption at fc,
        via ridge least squares in spline coefficient space.
        """
        # grid in normalized range space
        x_grid = np.linspace(0.0, 1.0, self.cfg.physics_init_grid_n, dtype=np.float64)
        r_phys = x_grid * (self.r_max - self.r_min) + self.r_min

        # target = -20 log10(r) - alpha(fc) * r_km
        target = -20.0 * np.log10(r_phys)
        if self.cfg.use_absorption:
            alpha_fc = float(thorp_absorption_db_per_km(np.array([self.cfg.fc_hz]))[0])  # dB/km
            target = target - alpha_fc * (r_phys / 1000.0)

        # gauge: subtract mean, store it (added back at inference)
        mean_tl = float(np.mean(target))
        # mean_tl is stored as a 0-dim buffer tensor; use copy_ for assignment.
        self.mean_tl.copy_(torch.tensor(mean_tl, dtype=self.mean_tl.dtype, device=self.mean_tl.device))
        target_centered = target - mean_tl

        stats = self.phi_r.set_parameters_from_least_squares(
            x_grid=x_grid, y_target=target_centered, ridge_lambda=self.cfg.spline.ridge_lambda
        )
        stats.update({"mean_tl": mean_tl})
        return stats


    @torch.no_grad()
    def gauge_fix(self, grid_n: Optional[int] = None) -> Dict[str, float]:
        """Fix the additive gauge by enforcing zero-mean components.

        The additive decomposition in Eq. (2) is not identifiable up to constant shifts:
        adding a constant to one component and subtracting it from ``bias`` leaves
        predictions unchanged. For physically meaningful inspection of each component,
        we enforce (approximately) zero-mean branches on a reference grid and absorb
        the removed constants into ``bias``.

        Parameters
        ----------
        grid_n : Optional[int]
            Number of points used in the reference grid per variable. If None,
            uses ``self.cfg.gauge_fix_grid_n``.

        Returns
        -------
        stats : Dict[str, float]
            Mean shifts removed from each branch.
        """
        n = int(self.cfg.gauge_fix_grid_n if grid_n is None else grid_n)
        device = self.bias.device
        dtype = self.bias.dtype

        r_grid = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1, 1)
        f_grid = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1, 1)

        mu_r = self.phi_r(r_grid).mean()
        mu_f = self.phi_f(f_grid).mean()

        # Shift each component by subtracting its mean; compensate in bias.
        self.phi_r.base_linear.bias.add_(-mu_r)
        self.phi_f.base_linear.bias.add_(-mu_f)
        self.bias.add_(mu_r + mu_f)

        return {"mu_r": float(mu_r.item()), "mu_f": float(mu_f.item())}


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[:, 0:1]
        f = x[:, 1:2]
        y_hat = self.phi_r(r) + self.phi_f(f) + self.bias + self.cfg.SL_db + self.mean_tl
        return y_hat


class SmallMLP(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)