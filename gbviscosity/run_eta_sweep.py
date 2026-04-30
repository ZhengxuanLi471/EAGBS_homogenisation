# Sweeps σ_η (log-normal GB viscosity distribution width) and ω for one seed,
# recording total and per-bin dissipation at each (σ_η, ω) pair.
#
# Usage:
#   python run_eta_sweep.py --seed 1 --sigma-values 3.0 1.5 0.5 0.1
#   python run_eta_sweep.py --seed 1 --sigma-min 0.1 --sigma-max 3.0 --sigma-steps 10

from main import solve_rve, build_spaces
from meshes import MakeMesh
from physics import *
from ngsolve import *
import os
import argparse
import pandas as pd
import numpy as np
from mpi4py import MPI
import json
import matplotlib.pyplot as plt
from collections import defaultdict

SetNumThreads(8)

# --- physical parameters ---
NU = 0.35
MU = 1.0
MACRO_SCALE = 1e-3
LN_OMEGA_MIN = -10
LN_OMEGA_MAX = 10
OMEGA_SAMPLES = 100

# Number of bins for viscosity grouping
N_VISCOSITY_BINS = 10


# --- viscosity distribution ---

def generate_lognormal_viscosities(contact_pairs, sigma, geo_mean_eta=1.0, seed=None):
    """Draw log-normal η_gb values; geometric mean fixed at geo_mean_eta, width controlled by sigma."""
    if seed is not None:
        np.random.seed(seed)
    
    n_boundaries = len(contact_pairs)
    
    if sigma < 1e-10:
        # Essentially uniform: all boundaries have geo_mean_eta
        return {key: geo_mean_eta for key in contact_pairs.keys()}
    
    mu = np.log(geo_mean_eta)  # geometric mean = exp(μ) for log-normal
    Z = np.random.randn(n_boundaries)
    eta_values = np.exp(mu + sigma * Z)
    
    gb_viscosity_map = {}
    for i, key in enumerate(contact_pairs.keys()):
        gb_viscosity_map[key] = float(eta_values[i])
    
    return gb_viscosity_map


def bin_viscosities(gb_viscosity_map, n_bins=N_VISCOSITY_BINS):
    """Assign each boundary to a log-spaced viscosity bin."""
    eta_values = np.array(list(gb_viscosity_map.values()))
    eta_min = max(eta_values.min(), 1e-10)
    eta_max = eta_values.max()
    
    # Create log-spaced bins
    bin_edges = np.logspace(np.log10(eta_min) - 0.1, np.log10(eta_max) + 0.1, n_bins + 1)
    
    boundary_bins = {}
    bin_boundaries = defaultdict(list)
    
    for key, eta in gb_viscosity_map.items():
        bin_idx = np.searchsorted(bin_edges[1:], eta)
        bin_idx = min(bin_idx, n_bins - 1)
        boundary_bins[key] = bin_idx
        bin_boundaries[bin_idx].append(key)
    
    return bin_edges, boundary_bins, dict(bin_boundaries)


# --- dissipation ---

def compute_dissipation_by_boundary(gfu, mesh, contact_pairs, gb_tangent_indices,
                                    gb_viscosity_map, omega, area):
    """Return per-boundary and total dissipation from the FE solution."""
    boundary_dissipation = {}
    
    for (a, b), (_, right_name) in contact_pairs.items():
        boundary_expr = f"slide_{right_name}"
        region = mesh.Boundaries(boundary_expr)
        idx_re, idx_im = gb_tangent_indices[(a, b)]
        
        t_s_re = gfu.components[idx_re]
        t_s_im = gfu.components[idx_im]
        magnitude_sq_s = t_s_im * t_s_im + t_s_re * t_s_re
        
        viscosity_factor = gb_viscosity_map.get((a, b), 1.0)
        contrib = (1.0 / viscosity_factor) * float(
            Integrate(magnitude_sq_s, mesh, BND, definedon=region)
        )
        # Scale by eta_ratio=1.0 and normalize
        boundary_dissipation[(a, b)] = 0.5 * 1.0 * contrib / area / omega
    
    total_dissipation = sum(boundary_dissipation.values())
    return boundary_dissipation, total_dissipation


def compute_binned_dissipation(boundary_dissipation, bin_boundaries, n_bins):
    """Sum boundary dissipation values into each viscosity bin."""
    bin_dissipation = {}
    for bin_idx in range(n_bins):
        boundaries = bin_boundaries.get(bin_idx, [])
        bin_dissipation[bin_idx] = sum(
            boundary_dissipation.get(key, 0.0) for key in boundaries
        )
    return bin_dissipation


# --- sweep ---

def run_sigma_sweep(mesh, spaces, contact_pairs, outer_contact_pairs,
                    corner_penalty_label, gb_tangent_indices,
                    Gamma, sigma, geo_mean_eta, viscosity_seed,
                    output_dir, ln_omega_min, ln_omega_max, omega_samples,
                    seedname, load_tag):
    """Run frequency sweep for one σ_η value and save results."""
    print(f"\n{'─'*60}")
    print(f"σ_η = {sigma:.3f} (geometric mean η = {geo_mean_eta:.3f})")
    print(f"{'─'*60}")
    
    gb_viscosity_map = generate_lognormal_viscosities(
        contact_pairs, sigma, geo_mean_eta, seed=viscosity_seed
    )
    eta_values = np.array(list(gb_viscosity_map.values()))
    print(f"  η: min={eta_values.min():.4e}, max={eta_values.max():.4e}, log-std={np.std(np.log(eta_values)):.4f}")

    bin_edges, boundary_bins, bin_boundaries = bin_viscosities(
        gb_viscosity_map, N_VISCOSITY_BINS
    )
    print(f"  Bin populations: ", end="")
    for bi in range(N_VISCOSITY_BINS):
        print(f"bin{bi}:{len(bin_boundaries.get(bi, []))} ", end="")
    print()
    
    ln_omega = np.linspace(ln_omega_min, ln_omega_max, omega_samples)
    
    # Storage
    total_diss = []
    bin_diss_arrays = {bi: [] for bi in range(N_VISCOSITY_BINS)}
    
    for j in range(len(ln_omega)):
        omegai = np.exp(ln_omega[j])
        print(f"\r  [{j+1}/{len(ln_omega)}] ω = {omegai:.4e}", end="", flush=True)
        
        gfu, mesh, convergence = solve_rve(
            spaces, mesh, contact_pairs, outer_contact_pairs,
            Gamma, nu=NU, mu=MU, omega=omegai, solver='cg', rtol=1e-8,
            corner_bnd=corner_penalty_label,
            gb_viscosity_map=gb_viscosity_map,
        )
        
        attempts = 0
        while not convergence and attempts < 10:
            ln_omega[j] += 0.01
            omegai = np.exp(ln_omega[j])
            gfu, mesh, convergence = solve_rve(
                spaces, mesh, contact_pairs, outer_contact_pairs,
                Gamma, nu=NU, mu=MU, omega=omegai, solver='cg', rtol=1e-8,
                corner_bnd=corner_penalty_label,
                gb_viscosity_map=gb_viscosity_map,
            )
            attempts += 1
        
        area = float(Integrate(1, mesh, VOL))
        boundary_dissipation, total_dissipation = compute_dissipation_by_boundary(
            gfu, mesh, contact_pairs, gb_tangent_indices,
            gb_viscosity_map, omegai, area,
        )
        
        binned_diss = compute_binned_dissipation(
            boundary_dissipation, bin_boundaries, N_VISCOSITY_BINS
        )
        
        total_diss.append(total_dissipation)
        for bi in range(N_VISCOSITY_BINS):
            bin_diss_arrays[bi].append(binned_diss.get(bi, 0.0))
    
    print()  # Newline after progress
    
    # ── Save results ────────────────────────────────────────────────────────
    modulus_scale = 2.0 / (MACRO_SCALE ** 2)
    
    result_data = {
        "ln_omega": ln_omega,
        "omega": np.exp(ln_omega),
        "E_diss_total": total_diss,
        "C_imag_total": modulus_scale * np.array(total_diss),
    }
    
    # Add bin contributions
    for bi in range(N_VISCOSITY_BINS):
        result_data[f"E_diss_bin{bi}"] = bin_diss_arrays[bi]
        result_data[f"C_imag_bin{bi}"] = modulus_scale * np.array(bin_diss_arrays[bi])
    
    df = pd.DataFrame(result_data)
    
    sigma_str = f"{sigma:.4f}".replace(".", "p")
    csv_path = os.path.join(
        output_dir,
        f"{seedname}_sigma_{sigma_str}_{load_tag}.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    
    # Save viscosity distribution info
    visc_info = {
        "sigma": sigma,
        "geo_mean_eta": geo_mean_eta,
        "viscosity_seed": viscosity_seed,
        "bin_edges": bin_edges.tolist(),
        "bin_counts": [len(bin_boundaries.get(bi, [])) for bi in range(N_VISCOSITY_BINS)],
        "gb_viscosities": {f"{k[0]}_{k[1]}": v for k, v in gb_viscosity_map.items()},
        "boundary_bins": {f"{k[0]}_{k[1]}": int(v) for k, v in boundary_bins.items()},
    }
    
    visc_path = os.path.join(
        output_dir,
        f"{seedname}_sigma_{sigma_str}_viscosity_info.json"
    )
    with open(visc_path, "w") as f:
        json.dump(visc_info, f, indent=2)
    
    return mesh, df


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep log-normal η distribution width and compute attenuation spectra"
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Tessellation seed index (default: 1)"
    )
    parser.add_argument(
        "--viscosity-seed", type=int, default=42,
        help="Random seed for viscosity generation (default: 42)"
    )
    parser.add_argument(
        "--sigma-values", type=float, nargs="+",
        help="List of σ_η values to sweep (overrides min/max/steps)"
    )
    parser.add_argument(
        "--sigma-min", type=float, default=0.05,
        help="Minimum σ_η (near-uniform, default: 0.05)"
    )
    parser.add_argument(
        "--sigma-max", type=float, default=3.0,
        help="Maximum σ_η (2-3 orders of magnitude, default: 3.0)"
    )
    parser.add_argument(
        "--sigma-steps", type=int, default=8,
        help="Number of σ_η values to sweep (default: 8)"
    )
    parser.add_argument(
        "--geo-mean-eta", type=float, default=0.12,
        help="Geometric mean viscosity (kept fixed, default: 0.12 to center peaks at ln(omega)~5)"
    )
    parser.add_argument(
        "--load", type=str, default="shear",
        choices=["shear", "normal_x", "normal_y"],
        help="Macro loading direction (default: shear)"
    )
    parser.add_argument(
        "--omega-samples", type=int, default=OMEGA_SAMPLES,
        help=f"Number of omega sample points (default: {OMEGA_SAMPLES})"
    )
    parser.add_argument(
        "--ln-omega-min", type=float, default=LN_OMEGA_MIN,
        help=f"Minimum ln(omega) (default: {LN_OMEGA_MIN})"
    )
    parser.add_argument(
        "--ln-omega-max", type=float, default=LN_OMEGA_MAX,
        help=f"Maximum ln(omega) (default: {LN_OMEGA_MAX})"
    )
    parser.add_argument(
        "--output-dir", type=str, default="eta_sweep_results",
        help="Output directory (default: eta_sweep_results)"
    )
    parser.add_argument(
        "--tess-json", type=str, default="tessellation_output.json",
        help="Tessellation JSON file (default: tessellation_output.json)"
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ── Load tessellation data ──────────────────────────────────────────────
    with open(args.tess_json, "r") as f:
        data = json.load(f)
    
    seedname = f"seeds_{args.seed}"
    if seedname not in data:
        raise ValueError(f"{seedname} not found in tessellation JSON")
    
    pts, regions = data[seedname]
    num_grains = len(regions)
    print(f"Tessellation: {seedname}, {num_grains} regions")
    
    # ── Build mesh ──────────────────────────────────────────────────────────
    try:
        (
            _, _, mesh, _,
            contact_pairs,
            outer_contact_pairs,
            corner_bnd_label,
            _outer_core_labels,
            _grain_areas,
            _grain_classes,
            _gb_size_type,
        ) = MakeMesh(pts, regions, maxh=0.1, comm=MPI.COMM_WORLD, core_frac=0.01)
    except Exception as e:
        print(f"MakeMesh failed: {e}")
        return
    
    print(f"Number of grain boundaries: {len(contact_pairs)}")
    
    # Corner penalty label
    penalty_boundaries = []
    if corner_bnd_label:
        if isinstance(corner_bnd_label, (list, tuple, set)):
            penalty_boundaries.extend(name for name in corner_bnd_label if name)
        else:
            penalty_boundaries.append(corner_bnd_label)
    penalty_boundaries = list(dict.fromkeys(penalty_boundaries))
    corner_penalty_label = "|".join(penalty_boundaries) if penalty_boundaries else None
    
    # Build FE spaces
    spaces = build_spaces(mesh, contact_pairs, outer_contact_pairs,
                          order_bulk=2, order_gb=1)
    gb_tangent_indices = spaces[4]
    
    # ── Macro load tensor ───────────────────────────────────────────────────
    load_map = {
        "shear":    ((0, 1), (0, 0)),
        "normal_x": ((1, 0), (0, 0)),
        "normal_y": ((0, 0), (0, 1)),
    }
    Gamma = load_map[args.load]
    
    # ── Determine sigma values to sweep ─────────────────────────────────────
    if args.sigma_values:
        sigma_values = sorted(args.sigma_values, reverse=True)  # Large to small
    else:
        sigma_values = np.logspace(
            np.log10(args.sigma_max),
            np.log10(args.sigma_min),
            args.sigma_steps
        ).tolist()
    
    print(f"\nσ_η values to sweep: {sigma_values}")
    print(f"Geometric mean η (fixed): {args.geo_mean_eta}")
    print(f"Loading: {args.load}")
    
    # ── Run sweep for each sigma ────────────────────────────────────────────
    all_results = {}
    for sigma in sigma_values:
        mesh, df = run_sigma_sweep(
            mesh, spaces, contact_pairs, outer_contact_pairs,
            corner_penalty_label, gb_tangent_indices,
            Gamma, sigma, args.geo_mean_eta, args.viscosity_seed,
            args.output_dir, args.ln_omega_min, args.ln_omega_max,
            args.omega_samples, seedname, args.load
        )
        all_results[sigma] = df
    
    # ── Save summary ────────────────────────────────────────────────────────
    summary = {
        "seed": args.seed,
        "seedname": seedname,
        "load": args.load,
        "geo_mean_eta": args.geo_mean_eta,
        "viscosity_seed": args.viscosity_seed,
        "sigma_values": sigma_values,
        "n_boundaries": len(contact_pairs),
        "n_bins": N_VISCOSITY_BINS,
        "omega_range": [args.ln_omega_min, args.ln_omega_max],
        "omega_samples": args.omega_samples,
    }
    
    summary_path = os.path.join(args.output_dir, f"{seedname}_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Sweep complete. Results saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
