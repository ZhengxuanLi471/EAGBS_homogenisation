# =============================================================================
# Plot Attenuation Spectra and Binned Contributions for η Distribution Sweep
# -----------------------------------------------------------------------------
# Visualizes the evolution of attenuation spectra as the log-normal viscosity
# distribution width σ_η shrinks from broad (2-3 orders of magnitude) to
# narrow (Debye-like peak).
#
# Produces:
#   1. Overlay plot: C_imag(ω) for all σ values showing broad → narrow transition
#   2. Stacked area plots: Binned contributions for each σ value
#   3. Waterfall/3D visualization of spectrum evolution
#   4. Peak analysis: Height and width vs σ
#
# Usage:
#   python plot_eta_sweep.py --results-dir eta_sweep_results
#   python plot_eta_sweep.py --results-dir eta_sweep_results --seedname seeds_1
#
# Author: Zhengxuan Li
# =============================================================================

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Plot style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


def load_sweep_data(results_dir, seedname=None, load_tag="shear"):
    """
    Load all CSV results from a sweep.
    
    Returns
    -------
    data : dict
        Maps sigma -> DataFrame
    summary : dict
        Sweep metadata
    viscosity_info : dict
        Maps sigma -> viscosity distribution info
    """
    # Find summary file
    summary_files = glob.glob(os.path.join(results_dir, "*_sweep_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No sweep summary found in {results_dir}")
    
    if seedname:
        summary_file = os.path.join(results_dir, f"{seedname}_sweep_summary.json")
    else:
        summary_file = summary_files[0]
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
    
    seedname = summary.get("seedname", seedname or "seeds_1")
    sigma_values = summary.get("sigma_values", [])
    
    data = {}
    viscosity_info = {}
    
    for sigma in sigma_values:
        sigma_str = f"{sigma:.4f}".replace(".", "p")
        csv_path = os.path.join(results_dir, f"{seedname}_sigma_{sigma_str}_{load_tag}.csv")
        visc_path = os.path.join(results_dir, f"{seedname}_sigma_{sigma_str}_viscosity_info.json")
        
        if os.path.exists(csv_path):
            data[sigma] = pd.read_csv(csv_path)
        else:
            print(f"Warning: {csv_path} not found")
        
        if os.path.exists(visc_path):
            with open(visc_path, "r") as f:
                viscosity_info[sigma] = json.load(f)
    
    return data, summary, viscosity_info


def load_and_average_sweep_data(results_dir, seed_start=1, seed_end=100, load_tag="shear"):
    """
    Load and average CSV results over multiple seeds.
    
    Returns
    -------
    avg_data : dict
        Maps sigma -> DataFrame with averaged values and std
    summary : dict
        Sweep metadata from first seed
    seed_count : dict
        Maps sigma -> number of seeds successfully loaded
    """
    # Find sigma values from first available summary
    summary_files = glob.glob(os.path.join(results_dir, "seeds_*_sweep_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No sweep summaries found in {results_dir}")
    
    # Load first summary to get sigma values
    with open(summary_files[0], "r") as f:
        summary = json.load(f)
    
    sigma_values = summary.get("sigma_values", [])
    print(f"Sigma values to average: {sigma_values}")
    
    # Collect data for each sigma across seeds
    all_data = {sigma: [] for sigma in sigma_values}
    seed_count = {sigma: 0 for sigma in sigma_values}
    
    for seed in range(seed_start, seed_end + 1):
        seedname = f"seeds_{seed}"
        for sigma in sigma_values:
            sigma_str = f"{sigma:.4f}".replace(".", "p")
            csv_path = os.path.join(results_dir, f"{seedname}_sigma_{sigma_str}_{load_tag}.csv")
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    all_data[sigma].append(df)
                    seed_count[sigma] += 1
                except Exception as e:
                    print(f"Warning: Failed to load {csv_path}: {e}")
    
    # Average the data
    avg_data = {}
    for sigma in sigma_values:
        if not all_data[sigma]:
            print(f"Warning: No data for sigma={sigma}")
            continue
        
        n_seeds = len(all_data[sigma])
        print(f"  σ={sigma:.4f}: averaging {n_seeds} seeds")
        
        # Stack all dataframes
        ref_df = all_data[sigma][0].copy()
        columns_to_avg = [c for c in ref_df.columns if c not in ['ln_omega', 'omega']]
        
        # Compute mean and std
        stacked = {col: np.array([df[col].values for df in all_data[sigma]]) 
                   for col in columns_to_avg}
        
        avg_df = ref_df.copy()
        for col in columns_to_avg:
            avg_df[col] = np.mean(stacked[col], axis=0)
            avg_df[f"{col}_std"] = np.std(stacked[col], axis=0)
            avg_df[f"{col}_sem"] = np.std(stacked[col], axis=0) / np.sqrt(n_seeds)
        
        avg_data[sigma] = avg_df
    
    return avg_data, summary, seed_count


def plot_spectrum_overlay(data, summary, output_dir):
    """
    Plot C_imag(ln ω) for all σ values on one figure.
    Shows the transition from broad, low peak to narrow, tall Debye peak.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sigma_values = sorted(data.keys(), reverse=True)
    colors = cm.viridis(np.linspace(0, 1, len(sigma_values)))
    
    for sigma, color in zip(sigma_values, colors):
        df = data[sigma]
        ax.plot(df["ln_omega"], df["C_imag_total"],
                label=f"σ = {sigma:.2f}",
                color=color, linewidth=2)
    
    ax.set_xlabel(r"$\ln(\omega)$")
    ax.set_ylabel(r"$C''$ (Imaginary Modulus)")
    ax.set_title("Attenuation Spectrum Evolution: Broad → Narrow with Decreasing σ")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(
        "Large σ: Broad, low amplitude\n(distributed relaxation times)",
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    ax.annotate(
        "Small σ: Narrow, tall peak\n(Debye-like, single τ)",
        xy=(0.95, 0.95), xycoords='axes fraction',
        fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrum_overlay.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "spectrum_overlay.pdf"))
    plt.close(fig)
    print(f"Saved spectrum_overlay.png/pdf")


def plot_averaged_spectrum_overlay(data, summary, seed_count, output_dir):
    """
    Plot averaged C_imag(ln ω) for all σ values with error bands.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sigma_values = sorted(data.keys(), reverse=True)
    colors = cm.viridis(np.linspace(0, 1, len(sigma_values)))
    
    for sigma, color in zip(sigma_values, colors):
        df = data[sigma]
        n_seeds = seed_count.get(sigma, 1)
        
        mean = df["C_imag_total"]
        sem = df.get("C_imag_total_sem", df["C_imag_total"] * 0)  # SEM for error band
        
        ax.plot(df["ln_omega"], mean,
                label=f"σ = {sigma:.2f} (n={n_seeds})",
                color=color, linewidth=2)
        ax.fill_between(df["ln_omega"], mean - 2*sem, mean + 2*sem,
                        color=color, alpha=0.2)
    
    ax.set_xlabel(r"$\ln(\omega)$")
    ax.set_ylabel(r"$\langle C'' \rangle$ (Averaged Imaginary Modulus)")
    ax.set_title(f"Averaged Attenuation Spectrum (n={max(seed_count.values())} seeds)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(
        "Large σ: Broad, low amplitude\n(distributed relaxation times)",
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=10, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    ax.annotate(
        "Small σ: Narrow, tall peak\n(Debye-like, single τ)",
        xy=(0.95, 0.95), xycoords='axes fraction',
        fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrum_overlay_averaged.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "spectrum_overlay_averaged.pdf"))
    plt.close(fig)
    print(f"Saved spectrum_overlay_averaged.png/pdf")


def plot_averaged_binned_contributions(data, summary, seed_count, output_dir):
    """
    For each σ value, show stacked area plot of averaged binned contributions.
    """
    n_bins = summary.get("n_bins", 10)
    sigma_values = sorted(data.keys(), reverse=True)
    
    # Color map for bins (log-spaced viscosity)
    bin_colors = cm.plasma(np.linspace(0.1, 0.9, n_bins))
    
    n_sigma = len(sigma_values)
    n_cols = min(3, n_sigma)
    n_rows = (n_sigma + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                              squeeze=False)
    
    for idx, sigma in enumerate(sigma_values):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        df = data[sigma]
        n_seeds = seed_count.get(sigma, 1)
        
        # Stack bin contributions
        bin_data = []
        for bi in range(n_bins):
            col_name = f"C_imag_bin{bi}"
            if col_name in df.columns:
                bin_data.append(df[col_name].values)
            else:
                bin_data.append(np.zeros(len(df)))
        
        bin_data = np.array(bin_data)
        
        ax.stackplot(df["ln_omega"], bin_data,
                     colors=bin_colors, alpha=0.8,
                     labels=[f"Bin {i}" for i in range(n_bins)])
        ax.plot(df["ln_omega"], df["C_imag_total"], 'k--', linewidth=1.5, label="Total")
        
        ax.set_xlabel(r"$\ln(\omega)$")
        ax.set_ylabel(r"$\langle C'' \rangle$")
        ax.set_title(f"σ = {sigma:.2f} (n={n_seeds})")
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(sigma_values), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.suptitle(f"Averaged Binned Viscosity Contributions (n seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "binned_contributions_averaged.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "binned_contributions_averaged.pdf"))
    plt.close(fig)
    print(f"Saved binned_contributions_averaged.png/pdf")


def plot_binned_contributions(data, summary, viscosity_info, output_dir):
    """
    For each σ value, show stacked area plot of binned contributions.
    """
    n_bins = summary.get("n_bins", 10)
    sigma_values = sorted(data.keys(), reverse=True)
    
    # Color map for bins (log-spaced viscosity)
    bin_colors = cm.plasma(np.linspace(0.1, 0.9, n_bins))
    
    n_sigma = len(sigma_values)
    n_cols = min(3, n_sigma)
    n_rows = (n_sigma + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                              squeeze=False)
    
    for idx, sigma in enumerate(sigma_values):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        df = data[sigma]
        ln_omega = df["ln_omega"].values
        
        # Stack bin contributions
        bin_data = []
        for bi in range(n_bins):
            col_name = f"C_imag_bin{bi}"
            if col_name in df.columns:
                bin_data.append(df[col_name].values)
            else:
                bin_data.append(np.zeros_like(ln_omega))
        
        bin_data = np.array(bin_data)
        
        # Create stacked area plot
        ax.stackplot(ln_omega, bin_data, colors=bin_colors,
                     labels=[f"Bin {i}" for i in range(n_bins)],
                     alpha=0.8)
        
        # Overlay total
        ax.plot(ln_omega, df["C_imag_total"], 'k-', linewidth=1.5,
                label="Total", alpha=0.7)
        
        ax.set_xlabel(r"$\ln(\omega)$")
        ax.set_ylabel(r"$C''$")
        ax.set_title(f"σ = {sigma:.2f}")
        ax.grid(True, alpha=0.3)
        
        # Add viscosity range annotation
        if sigma in viscosity_info:
            vinfo = viscosity_info[sigma]
            bin_counts = vinfo.get("bin_counts", [])
            total_boundaries = sum(bin_counts)
            ax.annotate(f"N = {total_boundaries}",
                       xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=9, ha='left', va='top')
    
    # Hide unused subplots
    for idx in range(n_sigma, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=Normalize(0, n_bins-1))
    sm.set_array([])
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "binned_contributions.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "binned_contributions.pdf"))
    plt.close(fig)
    print(f"Saved binned_contributions.png/pdf")


def plot_bin_contribution_evolution(data, summary, viscosity_info, output_dir):
    """
    Show how contributions from each viscosity bin evolve with σ.
    """
    n_bins = summary.get("n_bins", 10)
    sigma_values = sorted(data.keys(), reverse=True)
    
    # Find peak position and contributions at peak for each sigma
    peak_contributions = {bi: [] for bi in range(n_bins)}
    peak_totals = []
    
    for sigma in sigma_values:
        df = data[sigma]
        total = df["C_imag_total"].values
        peak_idx = np.argmax(total)
        peak_totals.append(total[peak_idx])
        
        for bi in range(n_bins):
            col_name = f"C_imag_bin{bi}"
            if col_name in df.columns:
                peak_contributions[bi].append(df[col_name].values[peak_idx])
            else:
                peak_contributions[bi].append(0.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Stacked bar chart of contributions at peak
    x = np.arange(len(sigma_values))
    width = 0.7
    bottom = np.zeros(len(sigma_values))
    
    bin_colors = cm.plasma(np.linspace(0.1, 0.9, n_bins))
    
    for bi in range(n_bins):
        heights = peak_contributions[bi]
        ax1.bar(x, heights, width, bottom=bottom, color=bin_colors[bi],
                label=f"Bin {bi}" if bi < 5 or bi > n_bins-3 else "")
        bottom += np.array(heights)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s:.2f}" for s in sigma_values], rotation=45)
    ax1.set_xlabel(r"$\sigma_\eta$")
    ax1.set_ylabel(r"$C''$ at peak")
    ax1.set_title("Contribution at Peak by Viscosity Bin")
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Right: Peak height and width vs sigma
    peak_heights = peak_totals
    
    # Estimate FWHM for each sigma
    fwhm_values = []
    for sigma in sigma_values:
        df = data[sigma]
        total = df["C_imag_total"].values
        ln_omega = df["ln_omega"].values
        
        max_val = total.max()
        half_max = max_val / 2
        
        # Find FWHM
        above_half = total > half_max
        if above_half.any():
            first_idx = np.argmax(above_half)
            last_idx = len(above_half) - 1 - np.argmax(above_half[::-1])
            fwhm = ln_omega[last_idx] - ln_omega[first_idx]
        else:
            fwhm = np.nan
        fwhm_values.append(fwhm)
    
    ax2_twin = ax2.twinx()
    
    l1 = ax2.plot(sigma_values, peak_heights, 'o-', color='C0',
                  linewidth=2, markersize=8, label="Peak Height")
    l2 = ax2_twin.plot(sigma_values, fwhm_values, 's--', color='C1',
                       linewidth=2, markersize=8, label="FWHM")
    
    ax2.set_xlabel(r"$\sigma_\eta$")
    ax2.set_ylabel(r"Peak Height ($C''_{max}$)", color='C0')
    ax2_twin.set_ylabel(r"FWHM in $\ln(\omega)$", color='C1')
    ax2.set_title("Peak Height and Width vs Distribution Width")
    
    ax2.tick_params(axis='y', labelcolor='C0')
    ax2_twin.tick_params(axis='y', labelcolor='C1')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "peak_evolution.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "peak_evolution.pdf"))
    plt.close(fig)
    print(f"Saved peak_evolution.png/pdf")


def load_bin_lengths(results_dir, seed_start=1, seed_end=100):
    """
    Load bin_total_lengths from viscosity_info JSONs across seeds.

    Returns
    -------
    length_data : dict
        sigma -> dict with keys 'bin_total_lengths' (n_bins array, averaged
        over seeds), 'bin_edges' (from first seed)
    """
    import glob as _glob
    visc_files = sorted(_glob.glob(
        os.path.join(results_dir, "seeds_*_sigma_*_viscosity_info.json")))
    if not visc_files:
        return None

    # Group by sigma value read from inside the JSON (avoids float precision
    # mismatches when parsing the filename).
    from collections import defaultdict
    sigma_groups = defaultdict(list)   # sigma_value -> list of file paths
    for vpath in visc_files:
        # Filter by seed range using the filename
        base = os.path.basename(vpath)
        parts = base.split("_sigma_")
        seed_str = parts[0]          # e.g. "seeds_42"
        seed_num = int(seed_str.replace("seeds_", ""))
        if seed_num < seed_start or seed_num > seed_end:
            continue
        # Read the exact sigma from the file contents (loaded later).
        # For grouping we still need a key now; use a rounded-filename key
        # temporarily and remap after reading the first file in each group.
        sigma_groups[vpath] = None    # placeholder

    # Now load files, group by exact sigma
    exact_groups = defaultdict(list)  # exact sigma -> list of vinfo dicts
    for vpath in sigma_groups:
        with open(vpath) as f:
            vinfo = json.load(f)
        if "bin_total_lengths" not in vinfo:
            continue
        exact_groups[vinfo["sigma"]].append(vinfo)

    length_data = {}
    for sigma, vinfos in sorted(exact_groups.items()):
        all_bin_lengths = []
        bin_edges = None
        n_bins = None
        for vinfo in vinfos:
            btl = vinfo["bin_total_lengths"]
            if n_bins is None:
                n_bins = len(btl)
                bin_edges = np.array(vinfo.get("bin_edges", []))
            arr = np.array([btl.get(str(bi), btl.get(bi, 0.0))
                            for bi in range(n_bins)])
            all_bin_lengths.append(arr)

        if not all_bin_lengths:
            continue
        stacked = np.array(all_bin_lengths)
        length_data[sigma] = {
            "bin_total_lengths_mean": np.mean(stacked, axis=0),
            "bin_total_lengths_std":  np.std(stacked, axis=0),
            "bin_edges": bin_edges,
            "n_seeds": len(all_bin_lengths),
        }
    return length_data if length_data else None


def plot_length_vs_dissipation(data, summary, results_dir, output_dir,
                               seed_start=1, seed_end=100):
    """
    Scatter / bar chart of total boundary length in each viscosity bin
    vs that bin's contribution to C'' at the peak frequency.

    Works for both single-seed and averaged data.
    """
    length_data = load_bin_lengths(results_dir, seed_start, seed_end)
    if length_data is None:
        print("  Skipping length-vs-dissipation plot – no bin_total_lengths in "
              "viscosity_info.  Run compute_gb_lengths.py first.")
        return

    n_bins = summary.get("n_bins", 10)
    sigma_values = sorted(data.keys(), reverse=True)

    # Determine number of rows/cols for subplots
    n_sigma = len(sigma_values)
    n_cols = min(n_sigma, 4)
    n_rows = int(np.ceil(n_sigma / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    bin_colors = cm.plasma(np.linspace(0.1, 0.9, n_bins))

    for idx, sigma in enumerate(sigma_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        if sigma not in length_data:
            ax.set_visible(False)
            continue

        ld = length_data[sigma]
        lengths = ld["bin_total_lengths_mean"]
        bin_edges = ld["bin_edges"]

        # Get C'' contributions at peak for this sigma
        df = data[sigma]
        total = df["C_imag_total"].values
        peak_idx = np.argmax(total)

        c_imag_bins = []
        for bi in range(n_bins):
            col_name = f"C_imag_bin{bi}"
            if col_name in df.columns:
                c_imag_bins.append(df[col_name].values[peak_idx])
            else:
                c_imag_bins.append(0.0)
        c_imag_bins = np.array(c_imag_bins)

        # Scatter plot: each bin is one point
        for bi in range(n_bins):
            if lengths[bi] < 1e-12 and c_imag_bins[bi] < 1e-15:
                continue
            label_str = ""
            if bi < len(bin_edges) - 1:
                label_str = f"[{bin_edges[bi]:.1e}, {bin_edges[bi+1]:.1e})"
            ax.scatter(lengths[bi], c_imag_bins[bi], s=80, color=bin_colors[bi],
                       edgecolors='k', linewidth=0.5, zorder=3, label=label_str)

        ax.set_xlabel("Total boundary length in bin")
        ax.set_ylabel(r"$C''_{\mathrm{bin}}$ at peak $\omega$")
        ax.set_title(rf"$\sigma_{{\eta}}$ = {sigma:.2f}")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_sigma, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Boundary Length in Bin vs Bin Dissipation at Peak", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "length_vs_dissipation.png"), dpi=200,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "length_vs_dissipation.pdf"),
                bbox_inches='tight')
    plt.close(fig)
    print("Saved length_vs_dissipation.png/pdf")


def plot_length_vs_integrated_dissipation(data, summary, results_dir, output_dir,
                                           seed_start=1, seed_end=100):
    """
    Scatter of total boundary length in each viscosity bin vs the *integrated*
    C'' contribution from that bin (area under the C''_bin curve over ln(ω)).
    """
    length_data = load_bin_lengths(results_dir, seed_start, seed_end)
    if length_data is None:
        print("  Skipping length-vs-integrated-dissipation plot – no bin_total_lengths. "
              "Run compute_gb_lengths.py first.")
        return

    n_bins = summary.get("n_bins", 10)
    sigma_values = sorted(data.keys(), reverse=True)

    n_sigma = len(sigma_values)
    n_cols = min(n_sigma, 4)
    n_rows = int(np.ceil(n_sigma / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    bin_colors = cm.plasma(np.linspace(0.1, 0.9, n_bins))

    for idx, sigma in enumerate(sigma_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        if sigma not in length_data:
            ax.set_visible(False)
            continue

        ld = length_data[sigma]
        lengths = ld["bin_total_lengths_mean"]
        bin_edges = ld["bin_edges"]

        df = data[sigma]
        ln_omega = df["ln_omega"].values

        # Integrate C''_bin over ln(ω) using the trapezoidal rule
        integrated_bins = []
        for bi in range(n_bins):
            col_name = f"C_imag_bin{bi}"
            if col_name in df.columns:
                integrated_bins.append(np.trapezoid(df[col_name].values, ln_omega))
            else:
                integrated_bins.append(0.0)
        integrated_bins = np.array(integrated_bins)

        for bi in range(n_bins):
            if lengths[bi] < 1e-12 and integrated_bins[bi] < 1e-15:
                continue
            label_str = ""
            if bi < len(bin_edges) - 1:
                label_str = f"[{bin_edges[bi]:.1e}, {bin_edges[bi+1]:.1e})"
            ax.scatter(lengths[bi], integrated_bins[bi], s=80, color=bin_colors[bi],
                       edgecolors='k', linewidth=0.5, zorder=3, label=label_str)

        ax.set_xlabel("Total boundary length in bin")
        ax.set_ylabel(r"$\int C''_{\mathrm{bin}}\,d\ln\omega$")
        ax.set_title(rf"$\sigma_{{\eta}}$ = {sigma:.2f}")
        ax.grid(True, alpha=0.3)

    for idx in range(n_sigma, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Boundary Length in Bin vs Integrated Bin Dissipation", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "length_vs_integrated_dissipation.png"), dpi=200,
                bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "length_vs_integrated_dissipation.pdf"),
                bbox_inches='tight')
    plt.close(fig)
    print("Saved length_vs_integrated_dissipation.png/pdf")


def plot_viscosity_distributions(viscosity_info, output_dir):
    """
    Plot the viscosity distributions for each σ value.
    """
    sigma_values = sorted(viscosity_info.keys(), reverse=True)
    
    n_sigma = len(sigma_values)
    fig, axes = plt.subplots(1, n_sigma, figsize=(3*n_sigma, 4), squeeze=False)
    
    for idx, sigma in enumerate(sigma_values):
        ax = axes[0, idx]
        vinfo = viscosity_info[sigma]
        
        eta_values = list(vinfo.get("gb_viscosities", {}).values())
        if not eta_values:
            continue
        
        eta_values = np.array(eta_values)
        
        # Histogram in log space
        log_eta = np.log10(eta_values)
        ax.hist(log_eta, bins=20, density=True, alpha=0.7, color='steelblue',
                edgecolor='white')
        
        ax.set_xlabel(r"$\log_{10}(\eta)$")
        ax.set_ylabel("Density")
        ax.set_title(f"σ = {sigma:.2f}")
        
        # Add statistics
        ax.axvline(np.mean(log_eta), color='red', linestyle='--',
                   label=f"Mean: {np.mean(log_eta):.2f}")
        ax.axvline(np.mean(log_eta) - np.std(log_eta), color='orange',
                   linestyle=':', alpha=0.7)
        ax.axvline(np.mean(log_eta) + np.std(log_eta), color='orange',
                   linestyle=':', alpha=0.7, label=f"±1σ")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "viscosity_distributions.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "viscosity_distributions.pdf"))
    plt.close(fig)
    print(f"Saved viscosity_distributions.png/pdf")


def plot_waterfall(data, summary, output_dir):
    """
    3D waterfall plot showing spectrum evolution with σ.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    sigma_values = sorted(data.keys(), reverse=True)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = cm.viridis(np.linspace(0, 1, len(sigma_values)))
    
    for i, (sigma, color) in enumerate(zip(sigma_values, colors)):
        df = data[sigma]
        ln_omega = df["ln_omega"].values
        c_imag = df["C_imag_total"].values
        
        # Create y values (sigma index)
        y = np.full_like(ln_omega, i)
        
        ax.plot(ln_omega, y, c_imag, color=color, linewidth=1.5)
        # Fill under curve
        ax.plot_surface(
            np.column_stack([ln_omega, ln_omega]),
            np.column_stack([y, y]),
            np.column_stack([np.zeros_like(c_imag), c_imag]),
            color=color, alpha=0.3
        )
    
    ax.set_xlabel(r'$\ln(\omega)$')
    ax.set_ylabel(r'$\sigma_\eta$ index')
    ax.set_zlabel(r"$C''$")
    ax.set_title('Attenuation Spectrum Evolution')
    
    # Set y-tick labels to sigma values
    ax.set_yticks(range(len(sigma_values)))
    ax.set_yticklabels([f"{s:.2f}" for s in sigma_values])
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "waterfall_3d.png"), dpi=200)
    plt.close(fig)
    print(f"Saved waterfall_3d.png")


def plot_normalized_spectra(data, summary, output_dir):
    """
    Plot normalized spectra (C''/C''_max) to emphasize shape change.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sigma_values = sorted(data.keys(), reverse=True)
    colors = cm.viridis(np.linspace(0, 1, len(sigma_values)))
    
    # Also plot ideal Debye peak for comparison
    ln_omega_debye = np.linspace(-10, 10, 200)
    omega_debye = np.exp(ln_omega_debye)
    # Debye: C'' ∝ ωτ / (1 + (ωτ)²)
    # Peak at ωτ = 1, so ln(ω) = -ln(τ)
    tau_debye = 1.0  # Centered at ln(ω) = 0
    debye_peak = omega_debye * tau_debye / (1 + (omega_debye * tau_debye)**2)
    debye_peak /= debye_peak.max()
    ax.plot(ln_omega_debye, debye_peak, 'k--', linewidth=2, alpha=0.5,
            label="Ideal Debye", zorder=0)
    
    for sigma, color in zip(sigma_values, colors):
        df = data[sigma]
        c_imag = df["C_imag_total"].values
        c_imag_norm = c_imag / c_imag.max()
        
        # Shift to align peaks at 0
        peak_idx = np.argmax(c_imag)
        ln_omega_shifted = df["ln_omega"].values - df["ln_omega"].values[peak_idx]
        
        ax.plot(ln_omega_shifted, c_imag_norm,
                label=f"σ = {sigma:.2f}",
                color=color, linewidth=2)
    
    ax.set_xlabel(r"$\ln(\omega) - \ln(\omega_{peak})$")
    ax.set_ylabel(r"$C'' / C''_{max}$ (Normalized)")
    ax.set_title("Normalized Attenuation Spectra (Peak-Aligned)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrum_normalized.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "spectrum_normalized.pdf"))
    plt.close(fig)
    print(f"Saved spectrum_normalized.png/pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from η distribution sweep"
    )
    parser.add_argument(
        "--results-dir", type=str, default="eta_sweep_results",
        help="Directory containing sweep results"
    )
    parser.add_argument(
        "--seedname", type=str, default=None,
        help="Tessellation seed name (e.g., seeds_1)"
    )
    parser.add_argument(
        "--load", type=str, default="shear",
        help="Loading direction tag (default: shear)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: results-dir/plots)"
    )
    parser.add_argument(
        "--average", action="store_true",
        help="Average results over multiple seeds (seeds_1 to seeds_100)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=1,
        help="First seed number for averaging (default: 1)"
    )
    parser.add_argument(
        "--seed-end", type=int, default=100,
        help="Last seed number for averaging (default: 100)"
    )
    args = parser.parse_args()
    
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    if args.average:
        # Load and average data over multiple seeds
        print(f"Loading and averaging data from {args.results_dir}...")
        print(f"Seeds: {args.seed_start} to {args.seed_end}")
        data, summary, seed_count = load_and_average_sweep_data(
            args.results_dir, args.seed_start, args.seed_end, args.load
        )
        
        print(f"Found {len(data)} sigma values: {sorted(data.keys(), reverse=True)}")
        
        # Generate averaged plots
        print("\nGenerating averaged plots...")
        
        plot_averaged_spectrum_overlay(data, summary, seed_count, output_dir)
        plot_averaged_binned_contributions(data, summary, seed_count, output_dir)
        plot_bin_contribution_evolution(data, summary, None, output_dir)
        plot_length_vs_dissipation(data, summary, args.results_dir, output_dir,
                                   args.seed_start, args.seed_end)
        plot_length_vs_integrated_dissipation(data, summary, args.results_dir, output_dir,
                                              args.seed_start, args.seed_end)
        plot_normalized_spectra(data, summary, output_dir)
        
        try:
            plot_waterfall(data, summary, output_dir)
        except Exception as e:
            print(f"Warning: 3D waterfall plot failed: {e}")
    else:
        # Load data for single seed
        print(f"Loading data from {args.results_dir}...")
        data, summary, viscosity_info = load_sweep_data(
            args.results_dir, args.seedname, args.load
        )
        
        print(f"Found {len(data)} sigma values: {sorted(data.keys(), reverse=True)}")
        
        # Generate plots
        print("\nGenerating plots...")
        
        plot_spectrum_overlay(data, summary, output_dir)
        plot_binned_contributions(data, summary, viscosity_info, output_dir)
        plot_bin_contribution_evolution(data, summary, viscosity_info, output_dir)
        plot_length_vs_dissipation(data, summary, args.results_dir, output_dir)
        plot_length_vs_integrated_dissipation(data, summary, args.results_dir, output_dir)
        plot_viscosity_distributions(viscosity_info, output_dir)
        plot_normalized_spectra(data, summary, output_dir)
        
        try:
            plot_waterfall(data, summary, output_dir)
        except Exception as e:
            print(f"Warning: 3D waterfall plot failed: {e}")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
