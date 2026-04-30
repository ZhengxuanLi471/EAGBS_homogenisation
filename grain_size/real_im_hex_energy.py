# =============================================================================
# Hexagonal Benchmark: Complex Stiffness vs Frequency
# -----------------------------------------------------------------------------
# - Builds the hexagonal tessellation, scales it to the grain area target,
#   and reuses the shared space/solver pipeline.
# - Sweeps ln(omega) for shear and normal load cases, generating VTU snapshots
#   CSV exports that act as regression baselines.
#
# Author: Zhengxuan Li
# Updated: 2 Dec 2025
# =============================================================================


from main import solve_rve,build_spaces
from meshes import MakeMesh
from physics import *
from ngsolve import *
import pandas as pd
import numpy as np
from mpi4py import MPI
import sys

SetNumThreads(32)

NU = 0.35
MU = 1.0
ETA_RATIO = 1.0  # eta0 / eta
MACRO_SCALE = 1e-3  # Gamma scaling applied in _setup_material_properties
LN_OMEGA_MIN = -3.0
LN_OMEGA_MAX = 10.0
OMEGA_SAMPLES = 100


def compute_energy_metrics(gfu, mesh, contact_pairs, gb_tangent_indices, omega, area, num_grains=6):
    """Return bulk storage/dissipation plus GB dissipation for one solution."""
    uR = gfu.components[0]
    uI = gfu.components[1]

    lam = 2 * MU * NU / (1 - 2 * NU)

    epsR = strain(uR)
    epsI = strain(uI)
    sigR = stress(uR, lam, MU)
    sigI = stress(uI, lam, MU)

    storage_density = InnerProduct(sigR, epsR) + InnerProduct(sigI, epsI)

    # Integrate grain by grain and sum
    bulk_storage_total = 0.0
    for grain_id in range(1, num_grains + 1):
        grain_region = mesh.Materials(f"region_{grain_id}")
        grain_storage = float(Integrate(storage_density, mesh, VOL, definedon=grain_region))
        bulk_storage_total += grain_storage
    
    bulk_storage = 0.5 * bulk_storage_total / area
    
    
    # Compute GB dissipation (tangential traction only), using explicit indices from build_spaces
    gb_energy = 0.0
    for (a, b), (_, right_name) in contact_pairs.items():
        boundary_expr = f"slide_{right_name}"
        region = mesh.Boundaries(boundary_expr)
        idx_re, idx_im = gb_tangent_indices[(a, b)]

        t_s_re = gfu.components[idx_re]
        t_s_im = gfu.components[idx_im]
        magnitude_sq_s = t_s_im * t_s_im + t_s_re * t_s_re
        gb_energy += float(Integrate(magnitude_sq_s, mesh, BND, definedon=region))

    gb_diss = 0.5 * ETA_RATIO * gb_energy / area/omega
    total_diss = gb_diss

    return bulk_storage, total_diss


def run_branch(Gamma, gamma_tag):
    """Sweep omega for one macro loading tensor and dump energy curves."""
    global mesh

    print("Starting with single Gamma:", Gamma)
    ln_omega = np.linspace(LN_OMEGA_MIN, LN_OMEGA_MAX, OMEGA_SAMPLES)

    storage_vals = []
    diss_total_vals = []
    diss_bulk_vals = []
    diss_gb_vals = []

    for j in range(len(ln_omega)):
        omegai = np.exp(ln_omega[j])
        print("Current omega: ", omegai)
        gfu, mesh, convergence = solve_rve(
            spaces,
            mesh,
            contact_pairs,
            outer_contact_pairs,
            Gamma,
            nu=NU,
            mu=MU,
            omega=omegai,
            solver='cg',
            rtol=1e-8,
            corner_bnd=corner_penalty_label,
        )
        print(convergence)
        while not convergence:
            ln_omega[j] += 0.01
            omegai = np.exp(ln_omega[j])
            print("Adjusting omega to ", omegai)
            gfu, mesh, convergence = solve_rve(
                spaces,
                mesh,
                contact_pairs,
                outer_contact_pairs,
                Gamma,
                nu=NU,
                mu=MU,
                omega=omegai,
                solver='cg',
                rtol=1e-8,
                corner_bnd=corner_penalty_label,
            )
            print(convergence)

        area = float(Integrate(1, mesh, VOL))
        storage, total_diss = compute_energy_metrics(
            gfu,
            mesh,
            contact_pairs,
            gb_tangent_indices,
            omegai,
            area,
        )
        storage_vals.append(storage)
        diss_total_vals.append(total_diss)

        # Compute energy density fields for VTU output
        uR = gfu.components[0]
        uI = gfu.components[1]
        lam = 2 * MU * NU / (1 - 2 * NU)
        epsR = strain(uR)
        epsI = strain(uI)
        sigR = stress(uR, lam, MU)
        sigI = stress(uI, lam, MU)
        storage_density = 0.5 * (InnerProduct(sigR, epsR) + InnerProduct(sigI, epsI))

        # Build dissipation density: project boundary traction data onto GridFunction for VTU output
        # Create GridFunction on H1 space to store dissipation density (zero in bulk)
        fes_scalar = H1(mesh, order=1)
        gf_diss = GridFunction(fes_scalar)
        gf_diss.vec[:] = 0  # Initialize to zero everywhere
        
        # For each GB, compute and add dissipation density on boundary DOFs
        for (a, b), (_, right_name) in contact_pairs.items():
            idx_re, idx_im = gb_tangent_indices[(a, b)]
            t_s_re = gfu.components[idx_re]
            t_s_im = gfu.components[idx_im]
            magnitude_sq_s = t_s_im * t_s_im + t_s_re * t_s_re
            gb_diss_density = 0.5 * ETA_RATIO * magnitude_sq_s / omegai
            boundary_region = mesh.Boundaries(f"slide_{right_name}")
            # Create temp GridFunction to hold this GB's contribution, then add to total
            gf_temp = GridFunction(fes_scalar)
            gf_temp.Set(gb_diss_density, definedon=boundary_region)
            gf_diss.vec.data += gf_temp.vec

        vtkout = VTKOutput(
            mesh,
            coefs=[gfu.components[0], gfu.components[1], storage_density, gf_diss],
            names=['real_deformation', 'imag_deformation', 'storage_density', 'dissipation_density'],
            filename="{}_{:.2f}".format(gamma_tag, ln_omega[j]),
            subdivision=2,
        )
        #vtkout.Do()

    omega_vals = np.exp(ln_omega)
    modulus_scale = 2.0 / (MACRO_SCALE ** 2)
    if gamma_tag == "shear":
        comp_name = "Cxyxy"
    elif gamma_tag == "normal_x":
        comp_name = "Cxxxx"
    else:
        comp_name = "Cyyyy"

    df = pd.DataFrame({
        'ln_omega': ln_omega,
        'omega': omega_vals,
        'E_storage': storage_vals,
        'E_diss_total': diss_total_vals,
        f'{comp_name}_real': modulus_scale * np.array(storage_vals),
        f'{comp_name}_imag': modulus_scale * np.array(diss_total_vals),
    })
    out_path = '{}_Seed_{}_modulus_data_{}.csv'.format(core_frac_tag, seedname, gamma_tag)
    df.to_csv(out_path, index=False)
    print(f"Saved energy curves to {out_path}")
 
a = np.sqrt(3)
pts0 = [
    (0, 0), (3/4, 0), (1/2, a/4), (0, a/4),
    (9/4, 0), (5/2, a/4), (2, 3*a/4), (1, 3*a/4),
    (3, 0), (3, a/4), (3, a), (9/4, a),
    (3/4, a), (0, a)
]
# Shift to center domain around origin
pts1 = [(x - 1.5, y - 0.5 * a) for (x, y) in pts0]
#scale to have average grain area of 1/50
area_per_grain = 3*np.sqrt(3)/2
scale_factor = np.sqrt(1/50 / area_per_grain)
pts = [(x * scale_factor, y * scale_factor) for (x, y) in pts1]

# Define polygonal regions (each grain) by vertex indices
regions = [
    (1, 2, 3, 4),
    (2, 5, 6, 7, 8, 3),
    (5, 9, 10, 6),
    (6, 10, 11, 12, 7),
    (8, 7, 12, 13),
    (4, 3, 8, 13, 14)
]

Gamma = ((0,1), (0, 0))
seedname = 'hex'

(
    _, _, mesh, _,
    contact_pairs,
    outer_contact_pairs,
    corner_bnd_label,
    _outer_core_labels,
 ) = MakeMesh(
    pts,
    regions,
    maxh=0.1*scale_factor,
    comm=MPI.COMM_WORLD,
     core_frac=(float(sys.argv[1]) if len(sys.argv) > 1 else 0.01) * scale_factor,
)

penalty_boundaries = []
if corner_bnd_label:
    if isinstance(corner_bnd_label, (list, tuple, set)):
        penalty_boundaries.extend(name for name in corner_bnd_label if name)
    else:
        penalty_boundaries.append(corner_bnd_label)
penalty_boundaries = list(dict.fromkeys(penalty_boundaries))
corner_penalty_label = "|".join(penalty_boundaries) if penalty_boundaries else None

spaces = build_spaces(
    mesh,
    contact_pairs,
    outer_contact_pairs,
    order_bulk=2,
    order_gb=1,
)
gb_tangent_indices = spaces[4]

# Tag used to prefix outputs based on CLI arg
core_frac_tag = (sys.argv[1] if len(sys.argv) > 1 else "0.01")

run_branch(((0,1), (0, 0)), "shear")
run_branch(((1,0), (0, 0)), "normal_x")
run_branch(((0,0), (0, 1)), "normal_y")
