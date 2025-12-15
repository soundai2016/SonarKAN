#!/usr/bin/env python
"""
Run the controlled surrogate experiment used for Fig. 2.

This script is intentionally self-contained:
- reads all hyperparameters from a YAML config
- trains SonarKAN (physics init) + SonarKAN (random init) + a parameter-matched MLP baseline
- optionally repeats over multiple random seeds and aggregates statistics
- writes all outputs under outputs/fig2/ (run-specific folder + stable figure path)

Usage
-----
python scripts/run_fig2_experiment.py --config configs/fig2.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# ---------------------------------------------------------------------
# Make `src/` importable without requiring users to set PYTHONPATH
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sonarkan.surrogate import AcousticSurrogate, AcousticSurrogateConfig  # noqa: E402
from sonarkan.models import SonarKAN, SonarKANConfig, BSplineLayerConfig, SmallMLP, count_parameters  # noqa: E402


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(nn.functional.mse_loss(pred, target)).item())


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient for 1D arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("pearson_corr: size mismatch")
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def epochs_to_within(loss: np.ndarray, frac: float = 0.05) -> int:
    """First epoch where loss <= (1+frac)*final_loss."""
    loss = np.asarray(loss, dtype=np.float64).reshape(-1)
    final = float(loss[-1])
    thr = final * (1.0 + float(frac))
    for i, v in enumerate(loss):
        if float(v) <= thr:
            return int(i)
    return int(len(loss) - 1)


def run_single(seed: int, cfg: dict) -> Dict[str, np.ndarray]:
    """Run one seed and return arrays for aggregation."""
    set_seeds(seed)

    # --- Surrogate (override seed for deterministic sampling per run) ---
    sur_cfg_dict = dict(cfg["surrogate"])
    sur_cfg_dict["seed"] = int(seed)
    sur_cfg = AcousticSurrogateConfig(**sur_cfg_dict)
    surrogate = AcousticSurrogate(sur_cfg)

    # --- Data (noiseless training/test for clean RMSE) ---
    n_train = int(cfg["training"]["n_train"])
    n_test = int(cfg["training"]["n_test"])
    X_train, y_train, _ = surrogate.generate_data(n_train, noise_std_db=0.0)
    X_test, y_test, _ = surrogate.generate_data(n_test, noise_std_db=0.0)

    # --- Models ---
    spline_cfg = BSplineLayerConfig(**cfg["model"]["spline"])
    sonarkan_cfg = SonarKANConfig(
        spline=spline_cfg,
        physics_init_grid_n=int(cfg["model"]["physics_init_grid_n"]),
        SL_db=float(cfg["model"]["SL_db"]),
        fc_hz=float(cfg["model"]["fc_hz"]),
        use_absorption=bool(cfg["model"]["use_absorption"]),
        gauge_fix_each_epoch=bool(cfg["model"].get("gauge_fix_each_epoch", True)),
        gauge_fix_grid_n=int(cfg["model"].get("gauge_fix_grid_n", 200)),
    )

    sonarkan_phys = SonarKAN(r_min_m=surrogate.r_min, r_max_m=surrogate.r_max, cfg=sonarkan_cfg)
    init_stats = sonarkan_phys.physics_init()

    sonarkan_rand = SonarKAN(r_min_m=surrogate.r_min, r_max_m=surrogate.r_max, cfg=sonarkan_cfg)
    # NOTE: no physics_init() call here (random init baseline for SonarKAN)

    mlp = SmallMLP(hidden=int(cfg["training"]["hidden_mlp"]))

    # --- Training (full-batch for transparency and deterministic reproduction) ---
    epochs = int(cfg["training"]["epochs"])
    lr = float(cfg["training"]["lr"])

    opt_phys = optim.Adam(sonarkan_phys.parameters(), lr=lr)
    opt_rand = optim.Adam(sonarkan_rand.parameters(), lr=lr)
    opt_mlp = optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_phys = np.zeros((epochs,), dtype=np.float64)
    loss_rand = np.zeros((epochs,), dtype=np.float64)
    loss_mlp = np.zeros((epochs,), dtype=np.float64)

    for ep in range(epochs):
        sonarkan_phys.train()
        sonarkan_rand.train()
        mlp.train()

        opt_phys.zero_grad()
        l_phys = criterion(sonarkan_phys(X_train), y_train)
        l_phys.backward()
        opt_phys.step()
        loss_phys[ep] = float(l_phys.item())
        if sonarkan_cfg.gauge_fix_each_epoch:
            sonarkan_phys.gauge_fix()

        opt_rand.zero_grad()
        l_rand = criterion(sonarkan_rand(X_train), y_train)
        l_rand.backward()
        opt_rand.step()
        loss_rand[ep] = float(l_rand.item())
        if sonarkan_cfg.gauge_fix_each_epoch:
            sonarkan_rand.gauge_fix()

        opt_mlp.zero_grad()
        l_mlp = criterion(mlp(X_train), y_train)
        l_mlp.backward()
        opt_mlp.step()
        loss_mlp[ep] = float(l_mlp.item())

    # --- Clean test RMSE ---
    sonarkan_phys.eval()
    sonarkan_rand.eval()
    mlp.eval()
    with torch.no_grad():
        rmse_phys_clean = rmse(sonarkan_phys(X_test), y_test)
        rmse_rand_clean = rmse(sonarkan_rand(X_test), y_test)
        rmse_mlp_clean = rmse(mlp(X_test), y_test)

    # --- SNR sweep (label noise) ---
    snr_levels = np.array(cfg["snr_sweep"]["levels_db"], dtype=np.float64)
    rmse_phys = np.zeros_like(snr_levels)
    rmse_rand = np.zeros_like(snr_levels)
    rmse_mlp_snr = np.zeros_like(snr_levels)

    sig_var = float(y_train.var().item())
    for i, snr in enumerate(snr_levels):
        noise_var = sig_var / (10.0 ** (snr / 10.0))
        noise_std = float(np.sqrt(noise_var))
        Xn, yn, _ = surrogate.generate_data(n_test, noise_std_db=noise_std)
        with torch.no_grad():
            rmse_phys[i] = rmse(sonarkan_phys(Xn), yn)
            rmse_rand[i] = rmse(sonarkan_rand(Xn), yn)
            rmse_mlp_snr[i] = rmse(mlp(Xn), yn)

    # --- Diagnostics for component recovery (reference grid) ---
    grid_n = int(cfg["diagnostics"].get("grid_n", 200))
    r_norm = torch.linspace(0.0, 1.0, grid_n).reshape(-1, 1)
    f_norm = torch.linspace(0.0, 1.0, grid_n).reshape(-1, 1)

    r_phys = (r_norm.numpy().reshape(-1) * (surrogate.r_max - surrogate.r_min) + surrogate.r_min)
    f_phys = (f_norm.numpy().reshape(-1) * (surrogate.f_max - surrogate.f_min) + surrogate.f_min)

    tl_base = surrogate.get_transmission_loss_base(r_phys)
    tl_full = surrogate.get_transmission_loss(r_phys)
    theory_r_base = -(tl_base)
    theory_r_full = -(tl_full)
    theory_r_base = theory_r_base - theory_r_base.mean()
    theory_r_full = theory_r_full - theory_r_full.mean()

    ts = surrogate.get_target_strength(f_phys)
    theory_f = ts - ts.mean()

    with torch.no_grad():
        learned_r_phys = sonarkan_phys.phi_r(r_norm).numpy().reshape(-1)
        learned_f_phys = sonarkan_phys.phi_f(f_norm).numpy().reshape(-1)
        learned_r_rand = sonarkan_rand.phi_r(r_norm).numpy().reshape(-1)
        learned_f_rand = sonarkan_rand.phi_f(f_norm).numpy().reshape(-1)

    # Center for inspection (gauge)
    learned_r_phys = learned_r_phys - learned_r_phys.mean()
    learned_f_phys = learned_f_phys - learned_f_phys.mean()
    learned_r_rand = learned_r_rand - learned_r_rand.mean()
    learned_f_rand = learned_f_rand - learned_f_rand.mean()

    comp_rmse_r_full_phys = float(np.sqrt(np.mean((learned_r_phys - theory_r_full) ** 2)))
    comp_rmse_r_full_rand = float(np.sqrt(np.mean((learned_r_rand - theory_r_full) ** 2)))
    comp_corr_r_full_phys = pearson_corr(learned_r_phys, theory_r_full)
    comp_corr_r_full_rand = pearson_corr(learned_r_rand, theory_r_full)

    comp_rmse_f_phys = float(np.sqrt(np.mean((learned_f_phys - theory_f) ** 2)))
    comp_rmse_f_rand = float(np.sqrt(np.mean((learned_f_rand - theory_f) ** 2)))
    comp_corr_f_phys = pearson_corr(learned_f_phys, theory_f)
    comp_corr_f_rand = pearson_corr(learned_f_rand, theory_f)

    # convergence-speed proxies
    ep_within5_phys = epochs_to_within(loss_phys, frac=0.05)
    ep_within5_rand = epochs_to_within(loss_rand, frac=0.05)
    ep_within5_mlp = epochs_to_within(loss_mlp, frac=0.05)

    return dict(
        # losses
        loss_sonarkan_phys=loss_phys,
        loss_sonarkan_rand=loss_rand,
        loss_mlp=loss_mlp,
        # clean rmse
        rmse_phys_clean=np.array([rmse_phys_clean], dtype=np.float64),
        rmse_rand_clean=np.array([rmse_rand_clean], dtype=np.float64),
        rmse_mlp_clean=np.array([rmse_mlp_clean], dtype=np.float64),
        # snr sweep rmse
        snr_levels=snr_levels,
        rmse_phys=rmse_phys,
        rmse_rand=rmse_rand,
        rmse_mlp_snr=rmse_mlp_snr,
        # components
        r_phys=r_phys,
        f_phys=f_phys,
        theory_r_base=theory_r_base,
        theory_r_full=theory_r_full,
        theory_f=theory_f,
        learned_r_phys=learned_r_phys,
        learned_f_phys=learned_f_phys,
        learned_r_rand=learned_r_rand,
        learned_f_rand=learned_f_rand,
        # init + params
        init_rmse=np.array([float(init_stats["rmse"])], dtype=np.float64),
        init_mean_tl=np.array([float(init_stats["mean_tl"])], dtype=np.float64),
        params_sonarkan=np.array([count_parameters(sonarkan_phys)], dtype=np.int64),
        params_mlp=np.array([count_parameters(mlp)], dtype=np.int64),
        # component metrics
        comp_rmse_r_full_phys=np.array([comp_rmse_r_full_phys], dtype=np.float64),
        comp_rmse_r_full_rand=np.array([comp_rmse_r_full_rand], dtype=np.float64),
        comp_corr_r_full_phys=np.array([comp_corr_r_full_phys], dtype=np.float64),
        comp_corr_r_full_rand=np.array([comp_corr_r_full_rand], dtype=np.float64),
        comp_rmse_f_phys=np.array([comp_rmse_f_phys], dtype=np.float64),
        comp_rmse_f_rand=np.array([comp_rmse_f_rand], dtype=np.float64),
        comp_corr_f_phys=np.array([comp_corr_f_phys], dtype=np.float64),
        comp_corr_f_rand=np.array([comp_corr_f_rand], dtype=np.float64),
        # convergence metrics
        ep_within5_phys=np.array([ep_within5_phys], dtype=np.int64),
        ep_within5_rand=np.array([ep_within5_rand], dtype=np.int64),
        ep_within5_mlp=np.array([ep_within5_mlp], dtype=np.int64),
    )


def stack_field(results: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([r[key] for r in results], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configs/fig2.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # seeds: either a single seed (cfg["seed"]) or a list (cfg["repeats"]["seeds"])
    if "repeats" in cfg and cfg["repeats"] and "seeds" in cfg["repeats"] and cfg["repeats"]["seeds"]:
        seeds = [int(s) for s in cfg["repeats"]["seeds"]]
    else:
        seeds = [int(cfg.get("seed", 42))]

    # --- Output run directory ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(cfg["outputs"]["root"])
    run_dir = root / f"run_{ts}"
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figures")

    # Save resolved config for reproducibility
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # --- Run ---
    per_seed: List[Dict[str, np.ndarray]] = []
    for seed in seeds:
        out_seed_dir = run_dir / f"seed_{seed}"
        ensure_dir(out_seed_dir)
        ensure_dir(out_seed_dir / "models")

        res = run_single(seed, cfg)
        per_seed.append(res)

        # (Optional) save per-seed results for debugging / re-plotting
        np.savez(out_seed_dir / "results_seed.npz", **res)

        # NOTE: we do not save per-seed model weights here to keep the artifact small.
        # If desired, users can uncomment the lines below to save weights.
        # torch.save(sonarkan_phys.state_dict(), out_seed_dir / "models" / "sonarkan_phys.pt")
        # torch.save(sonarkan_rand.state_dict(), out_seed_dir / "models" / "sonarkan_rand.pt")
        # torch.save(mlp.state_dict(), out_seed_dir / "models" / "mlp.pt")

    # --- Aggregate ---
    n_seeds = len(seeds)
    epochs = int(cfg["training"]["epochs"])
    snr_levels = per_seed[0]["snr_levels"]
    r_phys = per_seed[0]["r_phys"]
    f_phys = per_seed[0]["f_phys"]

    # stack arrays
    loss_sonarkan_phys = stack_field(per_seed, "loss_sonarkan_phys")
    loss_sonarkan_rand = stack_field(per_seed, "loss_sonarkan_rand")
    loss_mlp = stack_field(per_seed, "loss_mlp")

    rmse_phys_clean = np.concatenate([r["rmse_phys_clean"] for r in per_seed], axis=0)
    rmse_rand_clean = np.concatenate([r["rmse_rand_clean"] for r in per_seed], axis=0)
    rmse_mlp_clean = np.concatenate([r["rmse_mlp_clean"] for r in per_seed], axis=0)

    rmse_phys = stack_field(per_seed, "rmse_phys")
    rmse_rand = stack_field(per_seed, "rmse_rand")
    rmse_mlp_snr = stack_field(per_seed, "rmse_mlp_snr")

    learned_r_phys = stack_field(per_seed, "learned_r_phys")
    learned_f_phys = stack_field(per_seed, "learned_f_phys")
    learned_r_rand = stack_field(per_seed, "learned_r_rand")
    learned_f_rand = stack_field(per_seed, "learned_f_rand")

    # scalar diagnostics
    init_rmse = np.concatenate([r["init_rmse"] for r in per_seed], axis=0)
    init_mean_tl = np.concatenate([r["init_mean_tl"] for r in per_seed], axis=0)

    comp_rmse_r_full_phys = np.concatenate([r["comp_rmse_r_full_phys"] for r in per_seed], axis=0)
    comp_rmse_r_full_rand = np.concatenate([r["comp_rmse_r_full_rand"] for r in per_seed], axis=0)
    comp_corr_r_full_phys = np.concatenate([r["comp_corr_r_full_phys"] for r in per_seed], axis=0)
    comp_corr_r_full_rand = np.concatenate([r["comp_corr_r_full_rand"] for r in per_seed], axis=0)

    comp_rmse_f_phys = np.concatenate([r["comp_rmse_f_phys"] for r in per_seed], axis=0)
    comp_rmse_f_rand = np.concatenate([r["comp_rmse_f_rand"] for r in per_seed], axis=0)
    comp_corr_f_phys = np.concatenate([r["comp_corr_f_phys"] for r in per_seed], axis=0)
    comp_corr_f_rand = np.concatenate([r["comp_corr_f_rand"] for r in per_seed], axis=0)

    ep_within5_phys = np.concatenate([r["ep_within5_phys"] for r in per_seed], axis=0)
    ep_within5_rand = np.concatenate([r["ep_within5_rand"] for r in per_seed], axis=0)
    ep_within5_mlp = np.concatenate([r["ep_within5_mlp"] for r in per_seed], axis=0)

    # theory arrays from first run (identical across seeds by construction)
    theory_r_base = per_seed[0]["theory_r_base"]
    theory_r_full = per_seed[0]["theory_r_full"]
    theory_f = per_seed[0]["theory_f"]

    results_agg_path = run_dir / "results_aggregate.npz"
    np.savez(
        results_agg_path,
        seeds=np.array(seeds, dtype=np.int64),
        snr_levels=snr_levels,
        # losses
        loss_sonarkan_phys=loss_sonarkan_phys,
        loss_sonarkan_rand=loss_sonarkan_rand,
        loss_mlp=loss_mlp,
        # rmse (clean)
        rmse_phys_clean=rmse_phys_clean,
        rmse_rand_clean=rmse_rand_clean,
        rmse_mlp_clean=rmse_mlp_clean,
        # rmse (snr)
        rmse_phys=rmse_phys,
        rmse_rand=rmse_rand,
        rmse_mlp_snr=rmse_mlp_snr,
        # components
        r_phys=r_phys,
        f_phys=f_phys,
        theory_r_base=theory_r_base,
        theory_r_full=theory_r_full,
        theory_f=theory_f,
        learned_r_phys=learned_r_phys,
        learned_f_phys=learned_f_phys,
        learned_r_rand=learned_r_rand,
        learned_f_rand=learned_f_rand,
        # init + params
        init_rmse=init_rmse,
        init_mean_tl=init_mean_tl,
        params_sonarkan=per_seed[0]["params_sonarkan"],
        params_mlp=per_seed[0]["params_mlp"],
        # component metrics
        comp_rmse_r_full_phys=comp_rmse_r_full_phys,
        comp_rmse_r_full_rand=comp_rmse_r_full_rand,
        comp_corr_r_full_phys=comp_corr_r_full_phys,
        comp_corr_r_full_rand=comp_corr_r_full_rand,
        comp_rmse_f_phys=comp_rmse_f_phys,
        comp_rmse_f_rand=comp_rmse_f_rand,
        comp_corr_f_phys=comp_corr_f_phys,
        comp_corr_f_rand=comp_corr_f_rand,
        # convergence metrics
        ep_within5_phys=ep_within5_phys,
        ep_within5_rand=ep_within5_rand,
        ep_within5_mlp=ep_within5_mlp,
    )

    # human-readable summary
    def mean_std(x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0)

    summary = {
        "n_seeds": int(n_seeds),
        "seeds": seeds,
        "params_sonarkan": int(per_seed[0]["params_sonarkan"][0]),
        "params_mlp": int(per_seed[0]["params_mlp"][0]),
        "rmse_clean": {
            "sonarkan_phys_mean_std": mean_std(rmse_phys_clean),
            "sonarkan_rand_mean_std": mean_std(rmse_rand_clean),
            "mlp_mean_std": mean_std(rmse_mlp_clean),
        },
        "component_recovery": {
            "phi_r_rmse_full": {
                "sonarkan_phys_mean_std": mean_std(comp_rmse_r_full_phys),
                "sonarkan_rand_mean_std": mean_std(comp_rmse_r_full_rand),
            },
            "phi_r_corr_full": {
                "sonarkan_phys_mean_std": mean_std(comp_corr_r_full_phys),
                "sonarkan_rand_mean_std": mean_std(comp_corr_r_full_rand),
            },
            "phi_f_rmse": {
                "sonarkan_phys_mean_std": mean_std(comp_rmse_f_phys),
                "sonarkan_rand_mean_std": mean_std(comp_rmse_f_rand),
            },
            "phi_f_corr": {
                "sonarkan_phys_mean_std": mean_std(comp_corr_f_phys),
                "sonarkan_rand_mean_std": mean_std(comp_corr_f_rand),
            },
        },
        "convergence_speed": {
            "epochs_to_within_5pct_final_loss_mean_std": {
                "sonarkan_phys": mean_std(ep_within5_phys),
                "sonarkan_rand": mean_std(ep_within5_rand),
                "mlp": mean_std(ep_within5_mlp),
            }
        },
        "physics_init": {
            "ls_projection_rmse_mean_std": mean_std(init_rmse),
            "mean_tl_mean_std": mean_std(init_mean_tl),
        },
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Run dir: {run_dir}")
    print(f"[OK] Saved aggregate: {results_agg_path}")
    print(f"[OK] Saved summary: {run_dir / 'summary.json'}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Params SonarKAN={summary['params_sonarkan']}, MLP={summary['params_mlp']}")


if __name__ == "__main__":
    main()