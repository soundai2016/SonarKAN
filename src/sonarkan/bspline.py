
"""
B-spline utilities (NumPy + PyTorch)

This module provides:
- Open-uniform (clamped) knot construction on [xmin, xmax]
- Cox–de Boor recursion to evaluate B-spline basis functions

Notation:
- degree p >= 0 (cubic spline => p=3)
- number of basis functions n_basis >= p+1
- knot vector t has length n_basis + p + 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def make_open_uniform_knots(
    n_basis: int,
    degree: int,
    xmin: float = 0.0,
    xmax: float = 1.0,
) -> np.ndarray:
    """
    Construct an open-uniform (clamped) knot vector on [xmin, xmax].

    Parameters
    ----------
    n_basis : int
        Number of B-spline basis functions.
    degree : int
        Spline degree p (p=3 => cubic).
    xmin, xmax : float
        Domain endpoints.

    Returns
    -------
    knots : np.ndarray, shape (n_basis + degree + 1,)
    """
    if n_basis < degree + 1:
        raise ValueError(f"n_basis must be >= degree+1. Got n_basis={n_basis}, degree={degree}.")
    if xmax <= xmin:
        raise ValueError("xmax must be > xmin.")
    n_knots = n_basis + degree + 1
    # number of interior knots (excluding the repeated endpoints)
    n_inner = n_knots - 2 * (degree + 1)
    if n_inner < 0:
        raise RuntimeError("Invalid knot count derived from n_basis and degree.")
    if n_inner == 0:
        inner = np.array([], dtype=np.float64)
    else:
        inner = np.linspace(xmin, xmax, n_inner + 2, dtype=np.float64)[1:-1]
    knots = np.concatenate(
        [np.full(degree + 1, xmin, dtype=np.float64), inner, np.full(degree + 1, xmax, dtype=np.float64)]
    )
    return knots


def bspline_basis_matrix_np(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate all B-spline basis functions at x using Cox–de Boor recursion.

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (N,1)
    knots : np.ndarray, shape (n_basis + degree + 1,)
    degree : int

    Returns
    -------
    B : np.ndarray, shape (N, n_basis)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    n_basis = len(knots) - degree - 1
    N = x.shape[0]
    B = np.zeros((N, n_basis), dtype=np.float64)

    # degree-0 initialization
    for i in range(n_basis):
        left, right = knots[i], knots[i + 1]
        # Include the right endpoint only for the last basis to cover xmax.
        if i == n_basis - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        B[mask, i] = 1.0

    # recursion for higher degrees
    for d in range(1, degree + 1):
        B_new = np.zeros_like(B)
        for i in range(n_basis):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]

            term1 = 0.0
            if denom1 > 0:
                term1 = (x - knots[i]) / denom1 * B[:, i]

            term2 = 0.0
            if denom2 > 0 and i + 1 < n_basis:
                term2 = (knots[i + d + 1] - x) / denom2 * B[:, i + 1]

            B_new[:, i] = term1 + term2
        B = B_new
    return B


def bspline_basis_matrix_torch(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Torch version of bspline_basis_matrix_np.

    Parameters
    ----------
    x : torch.Tensor, shape (N,) or (N,1)
    knots : torch.Tensor, shape (n_basis + degree + 1,)
    degree : int

    Returns
    -------
    B : torch.Tensor, shape (N, n_basis)
    """
    if x.dim() == 2 and x.size(1) == 1:
        x = x[:, 0]
    elif x.dim() != 1:
        raise ValueError("x must have shape (N,) or (N,1).")

    knots = knots.reshape(-1)
    n_basis = knots.numel() - degree - 1
    N = x.numel()
    device = x.device
    dtype = x.dtype

    B = torch.zeros((N, n_basis), device=device, dtype=dtype)

    # degree-0 initialization
    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        if i == n_basis - 1:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        B[mask, i] = 1.0

    # recursion
    for d in range(1, degree + 1):
        B_new = torch.zeros_like(B)
        for i in range(n_basis):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]

            term1 = torch.zeros((N,), device=device, dtype=dtype)
            if float(denom1) > 0:
                term1 = (x - knots[i]) / denom1 * B[:, i]

            term2 = torch.zeros((N,), device=device, dtype=dtype)
            if float(denom2) > 0 and i + 1 < n_basis:
                term2 = (knots[i + d + 1] - x) / denom2 * B[:, i + 1]

            B_new[:, i] = term1 + term2
        B = B_new
    return B
