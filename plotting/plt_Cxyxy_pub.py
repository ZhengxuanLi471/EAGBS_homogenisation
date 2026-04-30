# Plots ensemble-averaged G'(ω) and G''(ω) across grain-size distribution widths.

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# File templates
SHEAR_TEMPLATE = 'Seed_seeds_{seed}_energy_real_im_data_shear.csv'
HEX_SHEAR = 'Seed_hex_energy_real_im_data_shear.csv'

# Base directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sigmas')

def discover_sigma_levels(base_dir: str):
    """Discover sigma-based folders."""
    levels = {}
    sigma_pattern = re.compile(r'^sigma_(\d*\.?\d+)$')
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        match = sigma_pattern.match(entry)
        if not match:
            continue
        try:
            sigma_value = float(match.group(1))
        except ValueError:
            continue
        levels[entry] = sigma_value
    return dict(sorted(levels.items(), key=lambda kv: (kv[1], kv[0])))


def detect_seeds(folder_path: str):
    """Detect available seed numbers in a folder."""
    pattern = re.compile(r'^Seed_seeds_(\d+)_')
    seeds = set()
    for fname in os.listdir(folder_path):
        match = pattern.match(fname)
        if match:
            seeds.add(int(match.group(1)))
    if not seeds:
        raise FileNotFoundError(f'No seed CSV files found in {folder_path}')
    return sorted(seeds)


# Discover sigma levels
ERROR_LEVELS = discover_sigma_levels(BASE_DIR)
ERROR_FOLDERS = list(ERROR_LEVELS.keys())

# Color scheme
viridis = mpl.colormaps['viridis']
color_positions = np.linspace(0.15, 0.85, len(ERROR_FOLDERS))
FOLDER_COLORS = {
    folder: viridis(pos)
    for folder, pos in zip(ERROR_FOLDERS, color_positions)
}

# ============================================================================
# Compute averaged data for each error level
# ============================================================================
folder_means = {}

for folder in ERROR_FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f'Skipping missing folder: {folder_path}')
        continue

    seeds = detect_seeds(folder_path)
    real_values = []
    imag_values = []
    ln_omega = None

    for seed in seeds:
        seed_path = os.path.join(folder_path, SHEAR_TEMPLATE.format(seed=seed))
        df = pd.read_csv(seed_path)
        if ln_omega is None:
            ln_omega = df['ln_omega'].to_numpy()
        real_values.append(df['Cxyxy_real'].to_numpy())
        imag_values.append(df['Cxyxy_imag'].to_numpy())

    folder_means[folder] = {
        'ln': ln_omega,
        'avg_real': np.mean(real_values, axis=0),
        'avg_imag': np.mean(imag_values, axis=0),
    }
    print(f'{folder} (σ={ERROR_LEVELS[folder]:.2f}): averaged {len(seeds)} seeds')

# Load hex reference from first available folder
hex_data = None
for folder in ERROR_FOLDERS:
    hex_path = os.path.join(BASE_DIR, folder, HEX_SHEAR)
    if os.path.isfile(hex_path):
        df_hex = pd.read_csv(hex_path)
        hex_data = {
            'ln': df_hex['ln_omega'].to_numpy(),
            'real': df_hex['Cxyxy_real'].to_numpy(),
            'imag': df_hex['Cxyxy_imag'].to_numpy(),
        }
        break

# ============================================================================
# Frequency normalization: ln(omega) -> ln(omega/omega_e)
# ============================================================================
# omega_e propto d_geom (geometric mean of effective diameter).
# For log-normal d with params (mu, sigma): d_geom = exp(mu).
# Constraint: E[d^2] = exp(2*mu + 2*sigma^2) = const across sigma levels,
# so mu = mu_0 - sigma^2  =>  d_geom(sigma) = d_0 * exp(-sigma^2).
# ln(omega_e(sigma)/omega_e_hex) = -sigma^2
#
# We set omega_e_hex so that the hex Im peak sits at ln(omega/omega_e) = 0:
#   ln_omega_e_hex = ln_omega at hex Im peak
# For each sigma level:
#   ln(omega/omega_e) = ln(omega) - ln_omega_e_hex + sigma^2

if hex_data is None:
    raise RuntimeError('Hex reference data not found; cannot normalise frequencies.')

# Parabolic interpolation to find sub-grid peak of Im part
def find_peak_parabolic(x, y):
    """Fit parabola to 3 points around argmax to get sub-grid peak position."""
    idx = np.argmax(y)
    if idx == 0 or idx == len(y) - 1:
        return x[idx]
    x0, x1, x2 = x[idx - 1], x[idx], x[idx + 1]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    # Vertex of parabola through (x0,y0), (x1,y1), (x2,y2)
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    return -b / (2 * a)

# Find hex Im peak position (parabolic interpolation)
ln_omega_e_hex = find_peak_parabolic(hex_data['ln'], hex_data['imag'])
print(f'\nHex Im peak at ln(omega) = {ln_omega_e_hex:.6f}  (parabolic)')

# Diagnostic: find empirical peak shifts and fit to C + alpha*sigma^2
sigmas_arr = []
empirical_shifts = []

print(f'\n{"Folder":<20} {"σ":>6} {"Peak ln(ω)":>14} {"Empirical Δ":>12} {"σ²":>8}')
print('-' * 66)
for folder in ERROR_FOLDERS:
    if folder not in folder_means:
        continue
    data = folder_means[folder]
    sigma = ERROR_LEVELS[folder]
    peak_ln = find_peak_parabolic(data['ln'], data['avg_imag'])
    emp_shift = ln_omega_e_hex - peak_ln
    sigmas_arr.append(sigma)
    empirical_shifts.append(emp_shift)
    print(f'{folder:<20} {sigma:>6.2f} {peak_ln:>14.6f} {emp_shift:>12.6f} {sigma**2:>8.4f}')

sigmas_arr = np.array(sigmas_arr)
empirical_shifts = np.array(empirical_shifts)

# Fit: empirical_shift = C + alpha * sigma^2
A = np.column_stack([np.ones_like(sigmas_arr), sigmas_arr**2])
(C_fit, alpha_fit), residuals, _, _ = np.linalg.lstsq(A, empirical_shifts, rcond=None)
print(f'\nFit: Δ = {C_fit:.6f} + {alpha_fit:.6f} * σ²')
print(f'  C (geometry disorder offset) = {C_fit:.6f}')
print(f'  α (σ² coefficient, expect 1) = {alpha_fit:.6f}')

# Apply normalization using theoretical sigma^2 shift only
# Convert to omega/omega_e ratio (not ln) for log-scale plotting
hex_data['omega_ratio'] = np.exp(hex_data['ln'] - ln_omega_e_hex)  # hex peak at 1

for folder in ERROR_FOLDERS:
    if folder not in folder_means:
        continue
    sigma = ERROR_LEVELS[folder]
    data = folder_means[folder]
    data['omega_ratio'] = np.exp(data['ln'] - ln_omega_e_hex + sigma**2)

# ============================================================================
# Publication figure: 2 subplots (a, b)
# ============================================================================
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 26,
    'legend.fontsize': 12,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

# --- Panel (a): Real part ---
for folder in ERROR_FOLDERS:
    if folder not in folder_means:
        continue
    data = folder_means[folder]
    color = FOLDER_COLORS[folder]
    sigma = ERROR_LEVELS[folder]
    ax_a.plot(data['omega_ratio'], data['avg_real'], linestyle='-', linewidth=1.5,
              color=color, label=f'$\\sigma_a = {sigma:.2f}$')

if hex_data is not None:
    ax_a.plot(hex_data['omega_ratio'], hex_data['real'], linestyle='--', linewidth=2,
              color='k', label='Hex')

ax_a.set_xscale('log')
ax_a.set_xlabel(r'$\omega/\omega_e$')
ax_a.set_ylabel(r"$G'/\mu$")
ax_a.legend(loc='best')
ax_a.grid(True, alpha=0.3)
ax_a.text(0.02, 0.95, 'a)', transform=ax_a.transAxes,
          fontsize=14, fontweight='bold', va='top')

# --- Panel (b): Imaginary part ---
for folder in ERROR_FOLDERS:
    if folder not in folder_means:
        continue
    data = folder_means[folder]
    color = FOLDER_COLORS[folder]
    sigma = ERROR_LEVELS[folder]
    ax_b.plot(data['omega_ratio'], data['avg_imag'], linestyle='-', linewidth=1.5,
              color=color, label=f'$\\sigma_a = {sigma:.2f}$')

if hex_data is not None:
    ax_b.plot(hex_data['omega_ratio'], hex_data['imag'], linestyle='--', linewidth=2,
              color='k', label='Hex')

ax_b.set_xscale('log')
ax_b.set_xlabel(r'$\omega/\omega_e$')
ax_b.set_ylabel(r"$G''/\mu$")
ax_b.legend(loc='best')
ax_b.grid(True, alpha=0.3)
ax_b.text(0.02, 0.95, 'b)', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', va='top')

fig.tight_layout()

# Save to manuscript_plots directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'Cxyxy_pub_real_imag.png')
output_pdf = os.path.join(script_dir, 'Cxyxy_pub_real_imag.pdf')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
fig.savefig(output_pdf, bbox_inches='tight')
print(f'\nSaved: {output_path}')
print(f'Saved: {output_pdf}')

plt.close(fig)
