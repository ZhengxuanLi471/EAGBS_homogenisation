# Fits the distributed-viscosity EAGBS reduced-order model to surface-wave
# velocity data (Priestley 2024) anchored to dry olivine experiments.

import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "mathtext.fontset": "cm",
})

# ---
# User settings
# ---
DATA_FILE = "keith.dat"

DELTA = 0.28
PERIOD_S = 40
D_MANTLE = 1e-3
D_REF = 5e-6
T_REF_C = 900.0
P_REF_GPA = 0.2
NPTS_DIST = 2001
N_SIGMA_DIST = 6

LOW_T_CUTOFF = 900.0
SIGMA_DEPTH = 50.0

# ---
# Constants
# ---
R_GAS = 8.314462618
RHO0 = 3213.0
ALPHA = 4.07e-5
KT_GPA = 115.0
G_MS2 = 9.81

# ---
# Parsing
# ---
_LINE_RE = re.compile(
    r"^\s*\d+\s+Keith depth=\s*([0-9.]+)\s*km,\s*T=\s*([0-9.]+)\s*Vs=([0-9.]+)\s*km s-1"
)

def read_keith_dat(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _LINE_RE.match(line)
            if m:
                depth_km, T_C, Vs_kms = map(float, m.groups())
                rows.append((depth_km, T_C, Vs_kms))
    if not rows:
        raise ValueError(f"No readable rows found in {path!r}")
    df = pd.DataFrame(rows, columns=["depth_km", "T_C", "Vs_kms"])
    return df.sort_values(["depth_km", "T_C"]).reset_index(drop=True)

# ---
# Self-consistent pressure from density
#
# rho(T, P) = RHO0 * (1 - ALPHA*T + P/KT_GPA)
# dP/dz = rho * g  ->  analytic solution:
#   P(z) = (a/b) * (exp(b*z) - 1)
# where a = RHO0*(1 - ALPHA*T)*g  [Pa/m]
#       b = RHO0*g / (KT_GPA*1e9) [1/m]
# ---
def pressure_from_depth_km(depth_km, T_C):
    z_m = np.asarray(depth_km, dtype=float) * 1e3
    T_C = np.asarray(T_C, dtype=float)
    a = RHO0 * (1.0 - ALPHA * T_C) * G_MS2
    b = RHO0 * G_MS2 / (KT_GPA * 1e9)
    return (a / b) * (np.exp(b * z_m) - 1.0) / 1e9  # GPa

# ---
# Model
# ---
def mu_GPa(T_C, P_GPa, mu0_GPa, dmu_dT_GPa_per_C, dmu_dP):
    return mu0_GPa + dmu_dT_GPa_per_C * np.asarray(T_C) + dmu_dP * np.asarray(P_GPa)

def rho(T_C, P_GPa):
    return RHO0 * (1.0 - ALPHA * np.asarray(T_C) + np.asarray(P_GPa) / KT_GPA)

def vs_unrelaxed(T_C, P_GPa, mu0_GPa, dmu_dT_GPa_per_C, dmu_dP):
    mu = np.maximum(mu_GPa(T_C, P_GPa, mu0_GPa, dmu_dT_GPa_per_C, dmu_dP), 1e-6)
    return np.sqrt(mu * 1e9 / rho(T_C, P_GPa)) / 1e3

def tau_eagbs(T_C, P_GPa, d_m, tau_ref_s, d_ref, T_ref_C, P_ref_GPa, E_Jmol,
              mu0_GPa, dmu_dT_GPa_per_C, dmu_dP):
    T_K = np.asarray(T_C) + 273.15
    T_ref_K = T_ref_C + 273.15
    mu_ref = mu_GPa(T_ref_C, P_ref_GPa, mu0_GPa, dmu_dT_GPa_per_C, dmu_dP)
    mu_now = np.maximum(mu_GPa(T_C, P_GPa, mu0_GPa, dmu_dT_GPa_per_C, dmu_dP), 1e-6)
    arrhenius = np.exp((E_Jmol / R_GAS) * (1.0 / T_K - 1.0 / T_ref_K))
    return tau_ref_s * (d_m / d_ref) * (mu_ref / mu_now) * (T_K / T_ref_K) * arrhenius

def lognormal_dist(sigma_ln, npts=NPTS_DIST, n_sigma=N_SIGMA_DIST):
    sigma_ln = float(sigma_ln)
    if sigma_ln <= 0.01:
        return np.array([1.0]), np.array([1.0])
    ln_tau = np.linspace(-n_sigma * sigma_ln, n_sigma * sigma_ln, npts)
    dln = ln_tau[1] - ln_tau[0]
    pdf = np.exp(-0.5 * (ln_tau / sigma_ln) ** 2) / (sigma_ln * np.sqrt(2.0 * np.pi))
    weights = pdf * dln
    weights /= weights.sum()
    return np.exp(ln_tau), weights

def Mstar_from_tau(omega, tau_center_s, sigma_ln, Delta):
    tau_hat, weights = lognormal_dist(sigma_ln)
    x = omega * tau_center_s * tau_hat
    return 1.0 - Delta * np.sum(weights / (1.0 + 1j * x))

def _compute_M(T_C, depth_km, params, period_s=PERIOD_S):
    mu0_GPa, dmu_dT, dmu_dP, log10_tau_ref, E_kJmol, sigma_ln = params
    P_GPa = pressure_from_depth_km(depth_km, T_C)
    omega = 2.0 * np.pi / period_s
    tau = tau_eagbs(T_C, P_GPa, D_MANTLE, 10.0 ** log10_tau_ref, D_REF,
                    T_REF_C, P_REF_GPA, E_kJmol * 1e3, mu0_GPa, dmu_dT, dmu_dP)
    return Mstar_from_tau(omega, tau, sigma_ln, DELTA)

def vs_dispersed_one(T_C, depth_km, params, period_s=PERIOD_S):
    mu0_GPa, dmu_dT, dmu_dP = params[0], params[1], params[2]
    P_GPa = pressure_from_depth_km(depth_km, T_C)
    M = _compute_M(T_C, depth_km, params, period_s)
    return vs_unrelaxed(T_C, P_GPa, mu0_GPa, dmu_dT, dmu_dP) * np.real(np.sqrt(M))

def qinv_one(T_C, depth_km, params, period_s=PERIOD_S):
    J = 1.0 / _compute_M(T_C, depth_km, params, period_s)
    return -np.imag(J) / np.real(J)

def vs_dispersed_vectorized(T_C, depth_km, params, period_s=PERIOD_S):
    T_C = np.asarray(T_C, dtype=float)
    depth_km = np.asarray(depth_km, dtype=float)
    out = np.empty_like(T_C)
    for i, (Ti, zi) in enumerate(zip(T_C, depth_km)):
        out[i] = vs_dispersed_one(Ti, zi, params, period_s)
    return out

def qinv_vectorized(T_C, depth_km, params, period_s=PERIOD_S):
    T_C = np.asarray(T_C, dtype=float)
    depth_km = np.asarray(depth_km, dtype=float)
    out = np.empty_like(T_C)
    for i, (Ti, zi) in enumerate(zip(T_C, depth_km)):
        out[i] = qinv_one(Ti, zi, params, period_s)
    return out

# ---
# Fitting
# ---
def residuals_full(x, df):
    pred = vs_dispersed_vectorized(df["T_C"].to_numpy(), df["depth_km"].to_numpy(), x)
    return pred - df["Vs_kms"].to_numpy()

def residuals_lowT(x3, df_lowT, x_fixed):
    x = np.array([x3[0], x3[1], x3[2], x_fixed[0], x_fixed[1], x_fixed[2]], dtype=float)
    pred = vs_dispersed_vectorized(df_lowT["T_C"].to_numpy(), df_lowT["depth_km"].to_numpy(), x)
    return pred - df_lowT["Vs_kms"].to_numpy()

def fit_model(df):
    x0 = np.array([72.4, -1.07e-2, 2.23, 4.15, 666.0, 3.5])
    lb = np.array([60.0, -3.0e-2, 0.0, 3.7, 400, 0.01])
    ub = np.array([90.0, -1.0e-4, 6.0, 5.1, 700, 8.0])

    df_lowT = df[(df["T_C"] <= LOW_T_CUTOFF) & (df["depth_km"] != 70.0)].copy()
    if len(df_lowT) >= 10:
        res1 = least_squares(residuals_lowT, x0[:3], bounds=(lb[:3], ub[:3]),
                             args=(df_lowT, x0[3:]), loss="soft_l1", f_scale=0.01, max_nfev=2000)
        x0[:3] = res1.x
    else:
        res1 = None

    df_highT = df[df["depth_km"] != 70.0].copy()
    res2 = least_squares(residuals_full, x0, bounds=(lb, ub), args=(df_highT,),
                         loss="soft_l1", f_scale=0.01, max_nfev=5000)
    return res1, res2

def summarize_fit(df, x_best):
    pred = vs_dispersed_vectorized(df["T_C"].to_numpy(), df["depth_km"].to_numpy(), x_best)
    resid = pred - df["Vs_kms"].to_numpy()
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))

    names = ["mu0_GPa", "dmu_dT_GPa_per_C", "dmu_dP", "log10_tau_ref_s", "E_kJmol", "sigma_ln"]
    print("\nBest-fit parameters")
    print("=" * 60)
    for name, val in zip(names, x_best):
        print(f"{name:>22s} = {val:.6g}")
    print(f"\nRMSE = {rmse:.6f} km/s")
    print(f"MAE  = {mae:.6f} km/s")

    tmp = df.copy()
    tmp["pred"] = pred
    tmp["resid"] = resid
    per_depth = {}
    for depth, g in tmp.groupby("depth_km"):
        rmse_d = np.sqrt(np.mean(g["resid"] ** 2))
        per_depth[depth] = rmse_d
        print(f"depth = {depth:5.1f} km : RMSE = {rmse_d:.6f} km/s")

    with open("fit_params_selfP.txt", "w") as f:
        f.write("Best-fit parameters (self-consistent pressure)\n")
        for name, val in zip(names, x_best):
            f.write(f"{name:>22s} = {val:.6g}\n")
        f.write(f"\nRMSE = {rmse:.6f} km/s\n")
        f.write(f"MAE  = {mae:.6f} km/s\n")
        for depth, rmse_d in per_depth.items():
            f.write(f"depth = {depth:5.1f} km : RMSE = {rmse_d:.6f} km/s\n")
    print("\nSaved fit_params_selfP.txt")

# ---
# Plot
# ---
def plot_fit(df, x_best, extend=False):
    T_max = 2500.0 if extend else df["T_C"].max()

    depth_styles = {
        40.0: ("k",       "o", "40 km"),
        50.0: ("tab:red", "s", "50 km"),
    }

    sigma_variants = [
        (0.01,      "tab:blue", "--", r"$\sigma_{\ln\tau}=0.01$"),
        (x_best[5], "k",        "-",  rf"$\sigma_{{\ln\tau}}={x_best[5]:.2f}$ (fit)"),
        (7.00,      "tab:red",  ":",  r"$\sigma_{\ln\tau}=7.00$"),
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

    # ---- Panel a: Vs vs T, all depths, best-fit ----
    for depth, (color, marker, dlabel) in depth_styles.items():
        g = df[df["depth_km"] == depth]
        if g.empty:
            continue
        ax_a.plot(g["T_C"], g["Vs_kms"], marker, ms=4, alpha=0.7, color=color, label=dlabel)
        T_grid = np.linspace(g["T_C"].min(), T_max, 400)
        z_grid = np.full_like(T_grid, depth)
        ax_a.plot(T_grid, vs_dispersed_vectorized(T_grid, z_grid, x_best),
                  "-", lw=2, color=color)

    ax_a.set_xlabel("Temperature (°C)")
    ax_a.set_ylabel(r"$V_s$ (km/s)")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend()
    ax_a.set_xlim(400, T_max)
    ax_a.set_ylim(4.2, 4.65)
    ax_a.text(0.03, 0.95, "a)", transform=ax_a.transAxes,
              va="top", fontweight="bold", fontsize=18)

    # ---- Panel b: Q^-1 vs T, SIGMA_DEPTH, 40–250 s period band ----
    PERIOD_MIN = 40.0
    PERIOD_MAX = 250.0

    T_grid_q = np.linspace(800, T_max, 400)
    z_grid_q = np.full_like(T_grid_q, SIGMA_DEPTH)

    for sigma, color, ls, slabel in sigma_variants:
        params = x_best.copy()
        params[5] = sigma
        Qi_lo = qinv_vectorized(T_grid_q, z_grid_q, params, period_s=PERIOD_MIN)
        Qi_hi = qinv_vectorized(T_grid_q, z_grid_q, params, period_s=PERIOD_MAX)
        ax_b.fill_between(T_grid_q, Qi_lo, Qi_hi, color=color, alpha=0.25, label=slabel)
        ax_b.semilogy(T_grid_q, Qi_lo, ls, lw=1, color=color)
        ax_b.semilogy(T_grid_q, Qi_hi, ls, lw=1, color=color)

    ax_b.axhspan(1e-2, 1e-1, color="gray", alpha=0.15)
    ax_b.set_xlabel("Temperature (°C)")
    ax_b.set_ylabel(r"$Q^{-1}$")
    ax_b.set_xlim(800, T_max)
    ax_b.set_ylim(1e-3, 0.3)
    ax_b.grid(True, alpha=0.3, which="both")
    ax_b.legend()
    ax_b.text(0.03, 0.95, "b)", transform=ax_b.transAxes,
              va="top", fontweight="bold", fontsize=18)

    plt.tight_layout()
    plt.savefig("fitted_selfP.png", dpi=300)
    print("\nSaved: fitted_selfP.png")

# ---
# Main
# ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", action="store_true",
                        help="Extend fitted curves to 2500 °C")
    args = parser.parse_args()

    df = read_keith_dat(DATA_FILE)
    print(df.head())
    print("\nCounts by depth:")
    print(df.groupby("depth_km").size())

    res1, res2 = fit_model(df)

    if res1 is not None:
        print("\nStage-1 low-T fit status:", res1.status, res1.message)

    print("\nStage-2 full fit status:", res2.status, res2.message)
    x_best = res2.x

    df_fit = df[df["depth_km"] != 70.0].reset_index(drop=True)
    summarize_fit(df_fit, x_best)
    plot_fit(df_fit, x_best, extend=args.extend)
