# Sonar Kolmogorov–Arnold Network，SonarKAN

This repository contains a **minimal, reproducible** reference implementation aligned with the
accompanying JASA Express Letters manuscript.

Design goals:

- **No hard-coded hyperparameters:** all settings are in YAML under `configs/`.
- **Experiment scripts are separated from plotting scripts:** `scripts/run_*` writes results; `scripts/plot_*` renders figures.
- **Reproducible file layout:** all artifacts are written under `outputs/` with:
  - a **run-specific** folder (timestamped), and
  - a **stable** figure path used by LaTeX (`outputs/fig1/...`, `outputs/fig2/...`).
- **Easy execution:** the scripts automatically add `src/` to `sys.path`
  (you do **not** need to set `PYTHONPATH`).

---

## 1) Environment

Recommended: Python **3.10–3.12**.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2) Reproduce Fig. 1 (mechanism diagram)

```bash
python scripts/draw_fig1.py --config configs/fig1.yaml
```

Output (stable path for LaTeX):
- `outputs/fig1/fig1_sonarkan_mechanism.png`

---

## 3) Reproduce Fig. 2 (controlled surrogate experiment)

### 3.1 Run the experiment

```bash
python scripts/run_fig2_experiment.py --config configs/fig2.yaml
```

This creates a run directory, e.g.:
- `outputs/fig2/run_YYYYMMDD_HHMMSS/`

Key outputs inside the run directory:
- `results_aggregate.npz` (all arrays needed for plotting)
- `summary.json` (mean±std metrics across seeds)
- `seed_*/results_seed.npz` (per-seed artifacts)

> Note: `configs/fig2.yaml` contains `repeats.seeds`. If you want a **single-run** figure,
> set `repeats.seeds: [42]` or remove the `repeats` block.

### 3.2 Plot Fig. 2

```bash
python scripts/plot_fig2.py --run_dir outputs/fig2/run_YYYYMMDD_HHMMSS --config configs/fig2.yaml
```

Outputs:
- `outputs/fig2/run_YYYYMMDD_HHMMSS/figures/fig2_sonarkan_simulation.png` (run-specific)
- `outputs/fig2/fig2_sonarkan_simulation.png` (stable path for LaTeX)

---

## 4) Repository layout

- `src/sonarkan/bspline.py`  
  Cox–de Boor B-spline basis implementation (NumPy + PyTorch).
- `src/sonarkan/models.py`  
  SonarKAN model, physics-informed initialization (ridge projection), and additive gauge-fixing.
- `src/sonarkan/surrogate.py`  
  Controlled sonar-equation-aligned surrogate used in Fig. 2.
- `scripts/run_fig2_experiment.py`  
  Runs the experiment and writes `results_aggregate.npz` + `summary.json`.
- `scripts/plot_fig2.py`  
  Renders the publication-quality Fig. 2 from `results_aggregate.npz`.

