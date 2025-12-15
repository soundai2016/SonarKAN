import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import yaml

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Make `src/` importable without requiring users to set PYTHONPATH
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sonarkan.bspline import make_open_uniform_knots, bspline_basis_matrix_np


def draw_sonarkan_mechanism(cfg: dict) -> str:
    out_path = cfg["outputs"]["path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Global font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2], height_ratios=[1, 0.8])

    # -------------------------
    # (a) Topological contrast
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("a  Topological Contrast: Entanglement vs. Disentanglement",
                  fontsize=14, fontweight="bold", loc="left", pad=20)
    ax1.axis("off")

    # MLP
    ax1.text(0.25, 0.88, "Standard MLP\n(Black Box)", ha="center", va="bottom",
             fontsize=12, fontweight="bold")
    inputs = [(0.1, 0.72), (0.1, 0.5), (0.1, 0.28)]
    hiddens = [(0.4, 0.78), (0.4, 0.64), (0.4, 0.5), (0.4, 0.36), (0.4, 0.22)]
    for inp in inputs:
        for hid in hiddens:
            ax1.plot([inp[0], hid[0]], [inp[1], hid[1]], color="gray", alpha=0.2, lw=0.8)
    for pos in inputs:
        ax1.add_patch(patches.Circle(pos, 0.035, facecolor="white", edgecolor="gray", lw=1.5))
    for pos in hiddens:
        ax1.add_patch(patches.Circle(pos, 0.035, facecolor="#e0e0e0", edgecolor="gray", lw=1.5))
    ax1.text(0.25, 0.1, "Global Matrix Multiplication\nFeatures Entangled",
             ha="center", fontsize=10, style="italic", color="#D55E00")

    ax1.plot([0.5, 0.5], [0.1, 0.9], color="black", linestyle="--", lw=1)

    # SonarKAN
    ax1.text(0.75, 0.88, "SonarKAN\n(Glass Box)", ha="center", va="bottom",
             fontsize=12, fontweight="bold")

    s_inputs = [(0.6, 0.72), (0.6, 0.5), (0.6, 0.28)]
    s_sum = (0.9, 0.5)

    labels = [r"$r$", r"$f$", r"$\theta$"]
    funcs = [r"$\phi_r$", r"$\phi_f$", r"$\phi_\theta$"]

    for i, inp in enumerate(s_inputs):
        x = np.linspace(inp[0], s_sum[0], 50)
        y = np.linspace(inp[1], s_sum[1], 50)
        if i == 1:
            curve = np.zeros_like(x)
        else:
            direction = 1 if i == 0 else -1
            curve = 0.06 * np.sin(np.linspace(0, np.pi, 50)) * direction
        ax1.plot(x, y + curve, color="#0072B2", lw=2.5, zorder=1)

        mid = 25
        ax1.text(x[mid], y[mid] + curve[mid] + (0.05 if i != 2 else -0.08),
                 funcs[i], color="#0072B2", fontsize=12, ha="center", fontweight="bold")

    for i, pos in enumerate(s_inputs):
        ax1.add_patch(patches.Circle(pos, 0.035, facecolor="white", edgecolor="#0072B2", lw=2, zorder=2))
        ax1.text(pos[0] - 0.05, pos[1], labels[i], ha="right", va="center", fontsize=12)

    ax1.add_patch(patches.Circle(s_sum, 0.045, facecolor="white", edgecolor="black", lw=2, zorder=2))
    ax1.text(s_sum[0], s_sum[1], r"$\Sigma$", ha="center", va="center", fontsize=14)
    ax1.text(0.75, 0.1, "Univariate Additivity\nInterpretable pathways",
             ha="center", fontsize=10, style="italic", color="#009E73")

    # -------------------------
    # (b) Micro mechanism
    # -------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("b  Micro-View: Learnable Edge Mechanism via B-splines",
                  fontsize=14, fontweight="bold", loc="left", pad=10)

    demo = cfg["spline_demo"]
    n_basis = int(demo["n_basis"])
    degree = int(demo["degree"])
    xmin, xmax = float(demo["xmin"]), float(demo["xmax"])

    x = np.linspace(xmin, xmax, 400)
    knots = make_open_uniform_knots(n_basis=n_basis, degree=degree, xmin=xmin, xmax=xmax)
    B = bspline_basis_matrix_np(x, knots, degree)  # (N, n_basis)

    # Coefficients: if provided fewer than n_basis, pad with zeros in the middle for clarity.
    coeffs_in = np.asarray(demo.get("coeffs", []), dtype=np.float64).reshape(-1)
    c = np.zeros((n_basis,), dtype=np.float64)
    if coeffs_in.size == 0:
        c[:] = np.linspace(0.6, -0.4, n_basis)
    else:
        # place in the central bases by default
        start = max(0, (n_basis - coeffs_in.size) // 2)
        end = min(n_basis, start + coeffs_in.size)
        c[start:end] = coeffs_in[:(end - start)]

    phi = (B * c.reshape(1, -1)).sum(axis=1)

    for i in range(n_basis):
        bi = B[:, i]
        ax2.plot(x, bi, color="gray", linestyle=":", alpha=0.5, lw=1)

        wi = c[i]
        fill_color = "#D55E00" if wi >= 0 else "#009E73"
        ax2.fill_between(x, 0, wi * bi, color=fill_color, alpha=0.15)

        # stem at the center of support [t_i, t_{i+degree+1}]
        left = knots[i]
        right = knots[i + degree + 1]
        center = 0.5 * (left + right)
        ax2.plot([center, center], [0, wi], color=fill_color, linestyle="--", alpha=0.6, lw=1.5)
        ax2.plot(center, wi, marker="o", color=fill_color, markersize=6, zorder=4)

    ax2.plot(x, phi, color="#0072B2", lw=3, label=r"Learned Function $\phi(x)$", zorder=5)
    ax2.text(0.05, 0.9, r"$\phi(x) = a x + c + \sum_i w_i\, B_i(x)$",
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.5"))

    ax2.set_xlabel("Input (normalized)")
    ax2.set_ylabel("Activation output")

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="#0072B2", lw=3),
        Line2D([0], [0], marker="o", color="#D55E00", linestyle="None"),
        Line2D([0], [0], color="gray", linestyle=":", lw=1),
    ]
    ax2.legend(custom_lines, ["Learned function", "Spline weights $w_i$", "B-spline bases $B_i(x)$"],
               loc="upper right", frameon=True, fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, linestyle=":", alpha=0.3)

    # -------------------------
    # (c) Isomorphism
    # -------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("c  Macro-View: Alignment to the Passive Sonar Equation",
                  fontsize=14, fontweight="bold", loc="left", pad=20)
    ax3.axis("off")

    bbox_props = dict(boxstyle="round,pad=0.6", fc="#f9f9f9", ec="black", lw=1.5)
    ax3.text(
        0.5, 0.82,
        r"$\mathbf{SE}= \mathbf{SL}(f)-\mathbf{TL}(r,f)-\mathbf{NL}(f)+\mathbf{DI}(\theta)-\mathbf{DT}$",
        ha="center", va="center", fontsize=16, bbox=bbox_props,
    )
    ax3.text(0.5, 0.66, r"(In our 2D simulation, $\mathbf{DI}$ is fixed and $\mathbf{DT}$ is absorbed into the bias.)",
             ha="center", va="center", fontsize=10, color="gray")

    start_x = 0.2
    gap = 0.3
    y_layer = 0.22
    
    # Adjusted dimensions for better spacing
    box_w = 0.26
    box_h = 0.24
    box_y = y_layer - 0.14

    terms = [
        (r"$\phi_f(f)$", r"$\approx \mathrm{SL}(f)-\mathrm{NL}(f)$", "#D55E00"),
        (r"$\phi_r(r)$", r"$\approx -\overline{\mathrm{TL}}(r)$", "#0072B2"),
        (r"$\phi_\theta(\theta)$", r"$\approx \mathrm{DI}(\theta)$", "#009E73"),
    ]

    for i, (term, desc, col) in enumerate(terms):
        x_pos = start_x + i * gap
        rect = patches.FancyBboxPatch(
            (x_pos - box_w / 2, box_y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05", fc="white", ec=col, lw=2.5
        )
        ax3.add_patch(rect)
        ax3.text(x_pos, y_layer + 0.03, term, ha="center", va="center",
                 fontsize=16, color=col, fontweight="bold")
        ax3.text(x_pos, y_layer - 0.07, desc, ha="center", va="center",
                 fontsize=10, color="#333333")
        ax3.annotate("", xy=(x_pos, 0.62), xytext=(x_pos, box_y + box_h + 0.04),
                     arrowprops=dict(arrowstyle="-|>", color="black", lw=2, mutation_scale=20))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configs/fig1.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_path = draw_sonarkan_mechanism(cfg)
    print(f"[OK] Fig.1 saved to: {out_path}")


if __name__ == "__main__":
    main()