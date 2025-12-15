#!/usr/bin/env python
"""
Plot Fig. 2 from a run directory produced by `run_fig2_experiment.py`.

Usage
-----
python scripts/plot_fig2.py --run_dir outputs/fig2/run_YYYYMMDD_HHMMSS --config configs/fig2.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml

# ---------------------------------------------------------------------
# Make `src/` importable without requiring users to set PYTHONPATH
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# -----------------------------
# Plot styling (JASA-friendly)
# -----------------------------
COLOR_SKAN_PHYS = "#0072B2"  # blue
COLOR_SKAN_RAND = "#009E73"  # green
COLOR_MLP = "#4D4D4D"        # dark gray
COLOR_THEORY = "#000000"     # black
COLOR_TS = "#D55E00"         # vermilion


def set_plot_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2.2


def clean_axes(ax):
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)
    ax.tick_params(direction="in", length=4)


def mean_std(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    if x.shape[axis] == 1:
        return np.mean(x, axis=axis), np.zeros_like(np.mean(x, axis=axis))
    return np.mean(x, axis=axis), np.std(x, axis=axis, ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory under outputs/fig2/")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/fig2.yaml (for labels)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results_agg = run_dir / "results_aggregate.npz"
    if not results_agg.exists():
        raise FileNotFoundError(f"Expected {results_agg}. Run the experiment script first.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = np.load(results_agg, allow_pickle=True)

    set_plot_style()

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    # --- Common arrays ---
    seeds = data["seeds"]
    n_seeds = int(seeds.size)

    r_phys = data["r_phys"]
    f_phys = data["f_phys"]
    theory_r_base = data["theory_r_base"]
    theory_r_full = data["theory_r_full"]
    theory_f = data["theory_f"]

    # learned components across seeds
    learned_r_phys = data["learned_r_phys"]
    learned_f_phys = data["learned_f_phys"]

    # --- Panel (a): Range component ---
    ax1 = fig.add_subplot(gs[0, 0])

    r_mean, r_std = mean_std(learned_r_phys, axis=0)

    ax1.plot(r_phys, theory_r_base, color=COLOR_THEORY, linestyle="--", alpha=0.7,
             label=r"Theory base: $-20\log_{10} r - \alpha(fc)\,r/1000$")
    ax1.plot(r_phys, theory_r_full, color=COLOR_THEORY, linestyle=":", alpha=0.6,
             label="Truth: full range term")
    ax1.plot(r_phys, r_mean, color=COLOR_SKAN_PHYS, alpha=0.95, label=r"SonarKAN (physics init) learned $\phi_r(r)$")
    if n_seeds > 1:
        ax1.fill_between(r_phys, r_mean - r_std, r_mean + r_std, alpha=0.15)

    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Component (dB, mean-centered)")
    ax1.set_title("a  Disentangling Transmission Loss Component", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="upper right")
    clean_axes(ax1)

    # --- Panel (b): Frequency component ---
    ax2 = fig.add_subplot(gs[0, 1])

    f_mean, f_std = mean_std(learned_f_phys, axis=0)

    ax2.plot(f_phys, theory_f, color=COLOR_TS, linestyle=":", linewidth=3.0, label="Truth: spectral signature term")
    ax2.plot(f_phys, f_mean, color=COLOR_SKAN_PHYS, label=r"SonarKAN learned $\phi_f(f)$")
    if n_seeds > 1:
        ax2.fill_between(f_phys, f_mean - f_std, f_mean + f_std, alpha=0.15)

    notch_center = float(cfg["surrogate"]["TS_notch_center_hz"])
    ax2.annotate("Resonance notch",
                 xy=(notch_center, float(np.min(theory_f))),
                 xytext=(notch_center - 700.0, float(np.min(theory_f) + 6.0)),
                 arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5,
                                 connectionstyle="arc3,rad=-0.2"),
                 fontsize=11)

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Component (dB, mean-centered)")
    ax2.set_title("b  Recovering Source Spectral Features", loc="left", fontweight="bold")
    ax2.legend(frameon=False, loc="lower left")
    clean_axes(ax2)

    # --- Panel (c): Training convergence ---
    ax3 = fig.add_subplot(gs[1, 0])

    loss_phys = data["loss_sonarkan_phys"]
    loss_rand = data["loss_sonarkan_rand"]
    loss_mlp = data["loss_mlp"]

    lphys_mean, lphys_std = mean_std(loss_phys, axis=0)
    lrand_mean, lrand_std = mean_std(loss_rand, axis=0)
    lmlp_mean, lmlp_std = mean_std(loss_mlp, axis=0)

    ax3.plot(lphys_mean, color=COLOR_SKAN_PHYS, label="SonarKAN (physics init)")
    ax3.plot(lrand_mean, color=COLOR_SKAN_RAND, linestyle="--", label="SonarKAN (random init)")
    ax3.plot(lmlp_mean, color=COLOR_MLP, label="MLP baseline")

    if n_seeds > 1:
        ax3.fill_between(np.arange(lphys_mean.size), lphys_mean - lphys_std, lphys_mean + lphys_std, alpha=0.10)
        ax3.fill_between(np.arange(lrand_mean.size), lrand_mean - lrand_std, lrand_mean + lrand_std, alpha=0.10)
        ax3.fill_between(np.arange(lmlp_mean.size), lmlp_mean - lmlp_std, lmlp_mean + lmlp_std, alpha=0.08)

    ax3.set_xlabel("Training epochs")
    ax3.set_ylabel("MSE loss (log scale)")
    ax3.set_yscale("log")
    ax3.set_title("c  Training Convergence Analysis", loc="left", fontweight="bold")
    ax3.legend(frameon=False, loc="upper right")
    clean_axes(ax3)

    # --- Panel (d): Robustness vs noise ---
    ax4 = fig.add_subplot(gs[1, 1])

    snr_levels = data["snr_levels"]
    rmse_phys = data["rmse_phys"]
    rmse_rand = data["rmse_rand"]
    rmse_mlp_snr = data["rmse_mlp_snr"]

    rp_mean, rp_std = mean_std(rmse_phys, axis=0)
    rr_mean, rr_std = mean_std(rmse_rand, axis=0)
    rm_mean, rm_std = mean_std(rmse_mlp_snr, axis=0)

    ax4.errorbar(snr_levels, rp_mean, yerr=rp_std if n_seeds > 1 else None,
                 color=COLOR_SKAN_PHYS, marker="o", markersize=7, label="SonarKAN (physics init)")
    ax4.errorbar(snr_levels, rr_mean, yerr=rr_std if n_seeds > 1 else None,
                 color=COLOR_SKAN_RAND, marker="^", markersize=7, linestyle="--", label="SonarKAN (random init)")
    ax4.errorbar(snr_levels, rm_mean, yerr=rm_std if n_seeds > 1 else None,
                 color=COLOR_MLP, marker="s", markersize=7, linestyle="-.", label="MLP baseline")

    ax4.set_xlabel("Label SNR (dB)")
    ax4.set_ylabel("Test RMSE (dB)")
    ax4.set_ylim(0, 50)
    ax4.set_title("d  Robustness Evaluation vs. Label Noise", loc="left", fontweight="bold")
    ax4.legend(frameon=False, loc="upper right")
    clean_axes(ax4)

    # Save outputs
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out1 = fig_dir / "fig2_sonarkan_simulation.png"
    fig.savefig(out1, dpi=300)

    # stable path for LaTeX inclusion
    stable_dir = Path(cfg["outputs"]["root"])
    stable_dir.mkdir(parents=True, exist_ok=True)
    out2 = stable_dir / "fig2_sonarkan_simulation.png"
    fig.savefig(out2, dpi=300)

    print(f"[OK] Saved: {out1}")
    print(f"[OK] Saved: {out2}")


if __name__ == "__main__":
    main()