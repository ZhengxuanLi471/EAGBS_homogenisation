# =============================================================================
# Compute Grain Boundary Lengths for Each Seed
# -----------------------------------------------------------------------------
# Builds the mesh from each tessellation seed, measures the length of every
# grain boundary (slide segment), and saves the results.  This is a lightweight
# post-processing step (no FE solve), so it runs quickly.
#
# Usage:
#   python compute_gb_lengths.py --seed 1
#   python compute_gb_lengths.py --seed 1 --seed-end 100   (loop over seeds)
#
# Author: Zhengxuan Li
# =============================================================================

from meshes import MakeMesh
from ngsolve import Integrate, BND
import os, json, argparse
import numpy as np
from mpi4py import MPI
from collections import defaultdict

# Must match run_eta_sweep.py
N_VISCOSITY_BINS = 10


def compute_boundary_lengths(mesh, contact_pairs):
    """Return dict  (i,j) -> float length  for every contact pair."""
    gb_lengths = {}
    for (a, b), (_, right_name) in contact_pairs.items():
        region = mesh.Boundaries(f"slide_{right_name}")
        length = float(Integrate(1, mesh, BND, definedon=region))
        gb_lengths[(a, b)] = length
    return gb_lengths


def bin_lengths(gb_lengths, gb_viscosity_map, bin_edges, n_bins=N_VISCOSITY_BINS):
    """Sum boundary lengths per viscosity bin.

    Returns
    -------
    bin_total_lengths : dict   bin_idx -> total length in that bin
    """
    eta_arr = np.array(list(gb_viscosity_map.values()))
    
    boundary_bins = {}
    for key, eta in gb_viscosity_map.items():
        boundary_bins[key] = int(np.searchsorted(bin_edges[1:], eta))

    bin_total_lengths = {}
    for bi in range(n_bins):
        total = sum(gb_lengths.get(k, 0.0)
                    for k, b in boundary_bins.items() if b == bi)
        bin_total_lengths[bi] = total
    return bin_total_lengths, boundary_bins


def process_seed(seed, tess_json, results_dir):
    """Build mesh for *seed*, compute boundary lengths, merge with viscosity info."""
    with open(tess_json) as f:
        data = json.load(f)

    seedname = f"seeds_{seed}"
    if seedname not in data:
        print(f"  {seedname} not in tessellation JSON – skipping")
        return

    pts, regions = data[seedname]
    print(f"  {seedname}: {len(regions)} grains … ", end="", flush=True)

    try:
        (_, _, mesh, _, contact_pairs,
         _outer, _corner, _oc, _ga, _gc, _gst
        ) = MakeMesh(pts, regions, maxh=0.1, comm=MPI.COMM_WORLD, core_frac=0.01)
    except Exception as e:
        print(f"MakeMesh failed: {e}")
        return

    gb_lengths = compute_boundary_lengths(mesh, contact_pairs)
    print(f"{len(gb_lengths)} boundaries, total L = {sum(gb_lengths.values()):.4f}")

    # --- For every sigma that was run, load viscosity_info, add lengths ----
    import glob
    visc_files = sorted(glob.glob(
        os.path.join(results_dir, f"{seedname}_sigma_*_viscosity_info.json")))

    for vpath in visc_files:
        with open(vpath) as f:
            vinfo = json.load(f)

        bin_edges = np.array(vinfo["bin_edges"])
        # Reconstruct gb_viscosity_map with tuple keys
        gb_visc = {}
        for k_str, v in vinfo["gb_viscosities"].items():
            a_str, b_str = k_str.split("_")
            gb_visc[(int(a_str), int(b_str))] = v

        btl, _ = bin_lengths(gb_lengths, gb_visc, bin_edges)

        # Per-boundary lengths
        per_boundary = {f"{k[0]}_{k[1]}": v for k, v in gb_lengths.items()}

        vinfo["gb_lengths"] = per_boundary
        vinfo["bin_total_lengths"] = {str(k): v for k, v in btl.items()}
        vinfo["total_boundary_length"] = sum(gb_lengths.values())

        with open(vpath, "w") as f:
            json.dump(vinfo, f, indent=2)

    print(f"    Updated {len(visc_files)} viscosity_info files")


def main():
    parser = argparse.ArgumentParser(
        description="Compute grain-boundary lengths and add to viscosity_info JSONs")
    parser.add_argument("--seed", type=int, default=1,
                        help="Starting seed (default: 1)")
    parser.add_argument("--seed-end", type=int, default=None,
                        help="Ending seed (inclusive). If omitted, only --seed is processed.")
    parser.add_argument("--tess-json", default="tessellation_output.json")
    parser.add_argument("--results-dir", default="eta_sweep_results")
    args = parser.parse_args()

    seed_end = args.seed_end or args.seed
    for s in range(args.seed, seed_end + 1):
        process_seed(s, args.tess_json, args.results_dir)

    print("Done.")


if __name__ == "__main__":
    main()
