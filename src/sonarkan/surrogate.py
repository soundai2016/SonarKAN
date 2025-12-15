"""
A lightweight physics surrogate for passive sonar-like measurements.

We generate synthetic data consistent with an (approximate) additive sonar equation
in the logarithmic (dB) domain:

    y = SL + TS(f) - TL(r, f) + ε

where:
- TL(r) includes spherical spreading, absorption at a fixed center frequency fc,
  and a range-localized multipath oscillation term (Lloyd's-mirror-like).
- TS(f) includes a resonant notch to emulate an elastic target spectral feature.

Default behavior matches the paper's controlled *separable* setting:
- TL depends on range only (absorption evaluated at a fixed center frequency fc),
- TS depends on frequency only.

To support stress-tests beyond strict separability, the surrogate optionally:
- introduces weak range--frequency coupling in the multipath residual, and/or
- evaluates absorption at the instantaneous frequency f (instead of a fixed fc), and/or
- performs incoherent (band-averaged) intensity averaging prior to conversion to dB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def thorp_absorption_db_per_km(f_hz: np.ndarray) -> np.ndarray:
    """
    Thorp absorption approximation (dB/km) for seawater.

    Parameters
    ----------
    f_hz : np.ndarray
        Frequency in Hz.

    Returns
    -------
    alpha_db_per_km : np.ndarray
        Absorption coefficient in dB/km.

    Notes
    -----
    Standard Thorp (approx) formula uses f in kHz:
      α = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2) + 2.75e-4 f^2 + 0.003
    """
    f_khz = np.asarray(f_hz, dtype=np.float64) / 1000.0
    f2 = f_khz ** 2
    alpha = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003
    return alpha


def db_to_lin(x_db: np.ndarray) -> np.ndarray:
    """Convert dB (power/intensity level) to linear scale."""
    x_db = np.asarray(x_db, dtype=np.float64)
    return 10.0 ** (x_db / 10.0)


def lin_to_db(x_lin: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Convert linear power/intensity to dB, with a small floor for numerical safety."""
    x_lin = np.asarray(x_lin, dtype=np.float64)
    x_lin = np.maximum(x_lin, float(floor))
    return 10.0 * np.log10(x_lin)


@dataclass
class AcousticSurrogateConfig:
    # Physical ranges
    r_min_m: float = 100.0
    r_max_m: float = 5000.0
    f_min_hz: float = 1000.0
    f_max_hz: float = 5000.0

    # Source level (dB re 1 μPa @ 1 m, abstracted)
    SL_db: float = 180.0

    # Target strength model (dB), resonance notch
    TS_base_db: float = 15.0
    TS_notch_depth_db: float = 12.0
    TS_notch_center_hz: float = 3500.0
    TS_notch_sigma_hz: float = 150.0

    # Transmission loss model components
    fc_hz: float = 3000.0
    multipath_amp_db: float = 4.0
    multipath_period_m: float = 450.0
    multipath_decay_m: float = 5000.0
    multipath_phase_rad: float = np.pi / 4

    # ------------------------------------------------------------
    # Optional: weak range--frequency coupling (stress-test mode)
    # ------------------------------------------------------------
    # If True, the multipath phase contains an additional (f-fc)*r term.
    use_rf_coupling: bool = False
    # Phase slope in rad per (kHz·km). Zero disables coupling even if use_rf_coupling=True.
    rf_phase_beta: float = 0.0
    # If True, absorption is evaluated at instantaneous f (instead of fixed fc),
    # introducing additional r--f coupling.
    absorption_depends_on_f: bool = False

    # ------------------------------------------------------------
    # Optional: incoherent band averaging (stress-test mode)
    # ------------------------------------------------------------
    # label_mode: "coherent" (default) or "incoherent_bandavg"
    label_mode: str = "coherent"
    bandavg_halfwidth_hz: float = 0.0
    bandavg_nfreq: int = 1

    # Random seed
    seed: int = 42


class AcousticSurrogate:
    def __init__(self, cfg: AcousticSurrogateConfig):
        self.cfg = cfg
        self.r_min = cfg.r_min_m
        self.r_max = cfg.r_max_m
        self.f_min = cfg.f_min_hz
        self.f_max = cfg.f_max_hz
        self.SL = cfg.SL_db
        np.random.seed(cfg.seed)

    def get_transmission_loss(self, r_m: np.ndarray, f_hz: Optional[np.ndarray] = None) -> np.ndarray:
        """
        TL(r,f) in dB: spherical spreading + absorption + oscillatory multipath residual.

        Notes
        -----
        - In the separable setting used for Fig. 2, absorption is evaluated at fixed fc
          and the multipath term depends only on r.
        - If requested, weak r--f coupling is introduced via a frequency-dependent
          multipath phase and/or frequency-dependent absorption.
        """
        r_m = np.asarray(r_m, dtype=np.float64)

        # If f is not provided, use fc for any optional frequency-dependent terms.
        if f_hz is None:
            f_hz = np.full_like(r_m, float(self.cfg.fc_hz), dtype=np.float64)
        else:
            f_hz = np.asarray(f_hz, dtype=np.float64)

        # Broadcast to a common shape (supports vectorized band-averaging).
        r_m, f_hz = np.broadcast_arrays(r_m, f_hz)

        # Spherical spreading (amplitude 1/r -> 20 log10 r in dB)
        tl_geo = 20.0 * np.log10(r_m)

        # Absorption (dB/km) * range(km)
        if self.cfg.absorption_depends_on_f:
            alpha = thorp_absorption_db_per_km(f_hz)  # same shape as inputs
        else:
            alpha = float(thorp_absorption_db_per_km(np.array([self.cfg.fc_hz]))[0])
        tl_abs = alpha * (r_m / 1000.0)

        # Range-localized oscillation (proxy for interference / eigenray beating)
        base_phase = 2.0 * np.pi * r_m / float(self.cfg.multipath_period_m) + float(self.cfg.multipath_phase_rad)
        if self.cfg.use_rf_coupling and float(self.cfg.rf_phase_beta) != 0.0:
            # Add a simple phase-mixing term proportional to (f-fc)*r (kHz·km scaling).
            f_khz = f_hz / 1000.0
            fc_khz = float(self.cfg.fc_hz) / 1000.0
            r_km = r_m / 1000.0
            base_phase = base_phase + float(self.cfg.rf_phase_beta) * (f_khz - fc_khz) * r_km

        mp = (
            float(self.cfg.multipath_amp_db)
            * np.sin(base_phase)
            * np.exp(-r_m / float(self.cfg.multipath_decay_m))
        )

        return tl_geo + tl_abs + mp

    def get_transmission_loss_base(self, r_m: np.ndarray) -> np.ndarray:
        """
        Base TL(r) used for physics-informed init: spherical spreading + absorption at fc.
        """
        r_m = np.asarray(r_m, dtype=np.float64)
        tl_geo = 20.0 * np.log10(r_m)
        alpha_fc = float(thorp_absorption_db_per_km(np.array([self.cfg.fc_hz]))[0])
        tl_abs = alpha_fc * (r_m / 1000.0)
        return tl_geo + tl_abs

    def get_target_strength(self, f_hz: np.ndarray) -> np.ndarray:
        f_hz = np.asarray(f_hz, dtype=np.float64)
        notch = self.cfg.TS_notch_depth_db * np.exp(-((f_hz - self.cfg.TS_notch_center_hz) ** 2) /
                                                   (2.0 * self.cfg.TS_notch_sigma_hz ** 2))
        return self.cfg.TS_base_db - notch

    def level_db_coherent(self, r_m: np.ndarray, f_hz: np.ndarray) -> np.ndarray:
        """Coherent (single-frequency) level in dB: y = SL - TL(r,f) + TS(f)."""
        TL = self.get_transmission_loss(r_m, f_hz=f_hz)
        TS = self.get_target_strength(f_hz)
        return self.SL - TL + TS

    def level_db_incoherent_bandavg(self, r_m: np.ndarray, f_center_hz: np.ndarray) -> np.ndarray:
        """
        Incoherent (band-averaged) level in dB.

        The averaging is performed on *linear* intensity:
            y_band = 10 log10( mean_k 10^{y_k/10} ).

        This provides a simple proxy for frequency-averaging that suppresses
        fine-scale interference microstructure when TL contains phase-mixing terms.
        """
        r_m = np.asarray(r_m, dtype=np.float64).reshape(-1)
        f_center_hz = np.asarray(f_center_hz, dtype=np.float64).reshape(-1)
        if r_m.shape[0] != f_center_hz.shape[0]:
            raise ValueError("level_db_incoherent_bandavg: r and f_center must have same length.")

        K = max(1, int(self.cfg.bandavg_nfreq))
        half = float(self.cfg.bandavg_halfwidth_hz)
        if K == 1 or half <= 0.0:
            return self.level_db_coherent(r_m, f_center_hz)

        offsets = np.linspace(-half, half, K, dtype=np.float64)  # (K,)
        f_grid = f_center_hz.reshape(-1, 1) + offsets.reshape(1, -1)  # (N,K)
        f_grid = np.clip(f_grid, self.f_min, self.f_max)

        r_grid = r_m.reshape(-1, 1)  # (N,1) broadcast with (N,K)
        y_grid = self.level_db_coherent(r_grid, f_grid)  # (N,K)

        I = db_to_lin(y_grid)
        I_mean = np.mean(I, axis=1)
        return lin_to_db(I_mean)

    def generate_data(self, n_samples: int, noise_std_db: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Generate normalized inputs X_norm (r,f) in [0,1]^2 and labels y in dB.
        Also returns physical inputs X_phys in original units for diagnostics.
        """
        r = np.random.uniform(self.r_min, self.r_max, n_samples)
        f = np.random.uniform(self.f_min, self.f_max, n_samples)

        mode = str(self.cfg.label_mode).strip().lower()
        if mode in ("coherent", "direct", "single"):
            y = self.level_db_coherent(r, f)
        elif mode in ("incoherent_bandavg", "bandavg", "incoherent"):
            y = self.level_db_incoherent_bandavg(r, f)
        else:
            raise ValueError(f"Unknown label_mode={self.cfg.label_mode!r}. Expected 'coherent' or 'incoherent_bandavg'.")

        if noise_std_db > 0.0:
            y = y + np.random.normal(0.0, noise_std_db, size=n_samples)

        X_phys = np.vstack([r, f]).T
        X_norm = np.zeros_like(X_phys, dtype=np.float64)
        X_norm[:, 0] = (r - self.r_min) / (self.r_max - self.r_min)
        X_norm[:, 1] = (f - self.f_min) / (self.f_max - self.f_min)

        return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(y.reshape(-1, 1), dtype=torch.float32), X_phys