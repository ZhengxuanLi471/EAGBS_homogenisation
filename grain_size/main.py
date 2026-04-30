# =============================================================================
# Complex EAGBS Solver
# -----------------------------------------------------------------------------
# - Constructs the coupled displacement/multiplier spaces for grain cores,
#   sliding interfaces, and periodic outer faces.
# - Assembles elasticity, grain-boundary, and macro-load contributions with
#   optional Nitsche penalties for anchoring periodic corners.
# - Solves the saddle system with Pardiso or CG (MUMPS-backed) and reports the
#   relative residual for convergence control.
# - Provides build_spaces(...) and solve_rve(...) entry points consumed by all
#   driver scripts in this repository.
#
# Author: Zhengxuan Li
# Updated: 2 Dec 2025
# =============================================================================

from ngsolve import *
import numpy as np
from math import sqrt
from mpi4py import MPI
from physics import *
from types import SimpleNamespace


# --- helpers ---------------------------------------------------------------

def _setup_material_properties(gamma, nu, mu):
    """Return the scaled conductivity tensor and Lamé parameter."""
    Gamma = CoefficientFunction(gamma, dims=(2, 2)) * 1e-3
    lam = 2 * mu * nu / (1 - 2 * nu)
    return Gamma, lam


def _core_boundary_names(boundary_info):
    """Collect unique core boundary names from the pairing metadata."""
    names = []
    if boundary_info:
        for name in boundary_info.get("core_names") or []:
            if name and name not in names:
                names.append(name)
    return names


def _core_boundary_expression(boundary_info):
    """Join core boundary names into a single Netgen expression."""
    names = _core_boundary_names(boundary_info)
    return "|".join(names)


def _add_gb_terms(a, mesh, sym, contact_pairs, omega, sliding=True):
    """Insert core/slide grain-boundary coupling blocks into the bilinear form."""
    for (i, j), (left_name, right_name) in contact_pairs.items():
        edge = f"{i}_{j}"
        region = "slide" if sliding else "core"
        cb = ContactBoundary(
            mesh.Boundaries(f"{region}_{right_name}"),
            mesh.Boundaries(f"{region}_{left_name}"),
            volume=False
        )

        r_s_Re = sym.__dict__[f"r_{edge}_s_Re"]
        r_s_Im = sym.__dict__[f"r_{edge}_s_Im"]
        t_s_Re = sym.__dict__[f"t_{edge}_s_Re"]
        t_s_Im = sym.__dict__[f"t_{edge}_s_Im"]

        r_n_Re = sym.__dict__[f"r_{edge}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{edge}_n_Im"]
        t_n_Re = sym.__dict__[f"t_{edge}_n_Re"]
        t_n_Im = sym.__dict__[f"t_{edge}_n_Im"]

        term1 = -complexify(
            (t_s_Re, t_s_Im),
            (sym.v_Re * specialcf.tangential(2) - sym.v_Re.Other() * specialcf.tangential(2),
             sym.v_Im * specialcf.tangential(2) - sym.v_Im.Other() * specialcf.tangential(2))
        )
        term2 = -complexify(
            (sym.u_Re * specialcf.tangential(2) - sym.u_Re.Other() * specialcf.tangential(2),
             sym.u_Im * specialcf.tangential(2) - sym.u_Im.Other() * specialcf.tangential(2)),
            (r_s_Re, r_s_Im)
        )

        if sliding:
            term3 = (1/omega) * complexify(
                (-t_s_Im, t_s_Re),
                (r_s_Re, r_s_Im)
            )
        else:
            term3 = 0

        term4 = -complexify(
            (t_n_Re, t_n_Im),
            (sym.v_Re * specialcf.normal(2) - sym.v_Re.Other() * specialcf.normal(2),
             sym.v_Im * specialcf.normal(2) - sym.v_Im.Other() * specialcf.normal(2))
        )
        term5 = -complexify(
            (sym.u_Re * specialcf.normal(2) - sym.u_Re.Other() * specialcf.normal(2),
             sym.u_Im * specialcf.normal(2) - sym.u_Im.Other() * specialcf.normal(2)),
            (r_n_Re, r_n_Im)
        )
        cb.AddIntegrator(term1 + term2 + term3 + term4 + term5 if sliding else term1 + term2 + term4 + term5)
        cb.Update(bf=a, intorder=12, maxdist=1e-9, both_sides=False)

def _add_outer_terms(
    a,
    mesh,
    sym,
    outer_contact_pairs,
    omega,
    sliding=True,
):
    """Impose periodic outer-boundary constraints via contact pairs."""
    if not outer_contact_pairs:
        return

    for key, data in outer_contact_pairs.items():
        minus_info = data["minus"]
        plus_info = data["plus"]
        minus_prefix = minus_info["prefix"]
        plus_prefix = plus_info["prefix"]
        edge = key
        region = "slide" if sliding else "core"
        #print('key=',key,' minus_prefix=',minus_prefix,' plus_prefix=',plus_prefix,' region=',region)

        displacement = data.get("displacement")
        disp_norm = None
        if displacement is not None:
            try:
                dx, dy = displacement
                disp_norm = float(np.hypot(float(dx), float(dy)))
            except Exception:
                disp_norm = None

        # Tighten pairing search radius to the expected periodic shift plus a small tolerance.
        maxdist_value = float(disp_norm + 1e-2) #if disp_norm is not None else 1e9
        #print('maxdist_value=',maxdist_value)

        r_s_Re = sym.__dict__[f"r_{edge}_s_Re"]
        r_s_Im = sym.__dict__[f"r_{edge}_s_Im"]
        t_s_Re = sym.__dict__[f"t_{edge}_s_Re"]
        t_s_Im = sym.__dict__[f"t_{edge}_s_Im"]

        r_n_Re = sym.__dict__[f"r_{edge}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{edge}_n_Im"]
        t_n_Re = sym.__dict__[f"t_{edge}_n_Re"]
        t_n_Im = sym.__dict__[f"t_{edge}_n_Im"]

        term1 = -complexify(
            (t_s_Re, t_s_Im),
            (sym.v_Re * specialcf.tangential(2) - sym.v_Re.Other() * specialcf.tangential(2),
                sym.v_Im * specialcf.tangential(2) - sym.v_Im.Other() * specialcf.tangential(2))
        )
        term2 = -complexify(
            (sym.u_Re * specialcf.tangential(2) - sym.u_Re.Other() * specialcf.tangential(2),
                sym.u_Im * specialcf.tangential(2) - sym.u_Im.Other() * specialcf.tangential(2)),
            (r_s_Re, r_s_Im)
        )

        term4 = -complexify(
            (t_n_Re, t_n_Im),
            (sym.v_Re * specialcf.normal(2) - sym.v_Re.Other() * specialcf.normal(2),
                sym.v_Im * specialcf.normal(2) - sym.v_Im.Other() * specialcf.normal(2))
        )
        term5 = -complexify(
            (sym.u_Re * specialcf.normal(2) - sym.u_Re.Other() * specialcf.normal(2),
                sym.u_Im * specialcf.normal(2) - sym.u_Im.Other() * specialcf.normal(2)),
            (r_n_Re, r_n_Im)
        )

        if region == "core":
            plus_names = _core_boundary_names(plus_info)
            minus_names = _core_boundary_names(minus_info)
            for plus_name, minus_name in zip(plus_names, minus_names):
                plus_region = mesh.Boundaries(plus_name)
                minus_region = mesh.Boundaries(minus_name)
                cb = ContactBoundary(plus_region, minus_region, volume=False)
                cb.AddIntegrator(term1 + term2 + term4 + term5)
                cb.Update(bf=a, intorder=12, maxdist=maxdist_value, both_sides=False)
        else:
            plus_region = mesh.Boundaries(f"{region}_{plus_prefix}")
            minus_region = mesh.Boundaries(f"{region}_{minus_prefix}")
            cb = ContactBoundary(
                plus_region,
                minus_region,
                volume=False
            )
            cb.AddIntegrator(term1 + term2 + term4 + term5)
            cb.Update(bf=a, intorder=12, maxdist=maxdist_value, both_sides=False)

def _assemble_bilinear_form(
    mesh,
    sym,
    contact_pairs,
    outer_contact_pairs,
    lam,
    mu,
    omega,
    fes,
):
    """Assemble elasticity, GB, and periodic contributions into one bilinear form."""
    a = BilinearForm(fes, check_unused=False)

    elastic_bf = complexify(
        (stress(sym.u_Re, lam, mu), stress(sym.u_Im, lam, mu)),
        (strain(sym.v_Re), strain(sym.v_Im))
    )
    a += elastic_bf * dx

    _add_gb_terms(a, mesh, sym, contact_pairs, omega, sliding=True)
    _add_gb_terms(a, mesh, sym, contact_pairs, omega, sliding=False)
    _add_outer_terms(
        a,
        mesh,
        sym,
        outer_contact_pairs,
        omega,
        sliding=True,
    )
    _add_outer_terms(
        a,
        mesh,
        sym,
        outer_contact_pairs,
        omega,
        sliding=False,
    )

    return a

def _add_corner_penalty(a, f, mesh, sym, CF_u, corner_bnd, gammaN=10):
    """Weakly anchor periodic corners to the macro field with a Nitsche-like term."""
    if not corner_bnd:
        return

    h = specialcf.mesh_size
    if isinstance(corner_bnd, str):
        boundary_names = [corner_bnd]
    else:
        try:
            boundary_names = [name for name in corner_bnd if name]
        except TypeError:
            boundary_names = [corner_bnd]

    for name in boundary_names:
        if not name:
            continue
        ds_region = ds(name)

        gamma_weight = (gammaN / h)

        # Bilinear: (gamma_weight) ∫ u_Re·v_Re
        a += gamma_weight * InnerProduct(sym.u_Re, sym.v_Re) * ds_region
        a += gamma_weight * InnerProduct(sym.u_Im, sym.v_Im) * ds_region

        # Linear: (gamma_weight) ∫ CF_u·v_Re
        f += gamma_weight * InnerProduct(CF_u, sym.v_Re) * ds_region

def _assemble_linear_form(mesh, sym, Gamma, fes, outer_contact_pairs):
    """Assemble load vector enforcing macro traction jumps on periodic faces."""
    f = LinearForm(fes)
    if not outer_contact_pairs:
        return f

    for key, data in outer_contact_pairs.items():
        plus_info = data["plus"]
        prefix = plus_info["prefix"]
        core_names = _core_boundary_names(plus_info)
        ds_region = core_names + [f"slide_{prefix}"]

        displacement = data.get("displacement")
        if displacement is None:
            raise ValueError(f"Missing displacement vector for outer contact pair {key}.")
        try:
            dx, dy = displacement
            disp = CoefficientFunction((float(dx), float(dy)))
        except Exception as exc:
            raise ValueError(f"Invalid displacement data for outer contact pair {key}: {displacement}") from exc
        strain_jump = Gamma * disp
        jump_normal = InnerProduct(strain_jump, specialcf.normal(2))
        jump_tangential = InnerProduct(strain_jump, specialcf.tangential(2))

        r_n_Re = sym.__dict__[f"r_{key}_n_Re"]
        r_n_Im = sym.__dict__[f"r_{key}_n_Im"]
        r_s_Re = sym.__dict__[f"r_{key}_s_Re"]
        r_s_Im = sym.__dict__[f"r_{key}_s_Im"]
        for name in ds_region:
            f += -jump_normal * r_n_Re * ds(name)
            f += -jump_normal * r_n_Im * ds(name)
            f += -jump_tangential * r_s_Re * ds(name)
            f += -jump_tangential * r_s_Im * ds(name)

    return f

def _solve(a, f, fes, solver, rtol):
    """Solve the saddle system with Pardiso or CG and report the relative residual."""
    from ngsolve.krylovspace import CGSolver

    if solver == "direct":
        with TaskManager():
            a.Assemble()
            f.Assemble()

            gfu = GridFunction(fes)
            gfu.vec.data = a.mat.Inverse(inverse="pardiso") * f.vec

            rel_residual = Norm(a.mat * gfu.vec.data - f.vec.data) / Norm(f.vec.data)
        print(f"Relative residual: {rel_residual:e}")
        convergence = bool(rel_residual < rtol)
        return gfu, convergence

    else:
        with TaskManager():
            a.Assemble()
            f.Assemble()
            c = Preconditioner(a, "multigrid")
            c.Update()

            gfu = GridFunction(fes)
            inv = CGSolver(mat=a.mat, pre=c.mat,
                           printrates='\r', maxiter=50, tol=rtol)
            gfu.vec.data = inv * f.vec

            rel_residual = Norm(a.mat * gfu.vec.data - f.vec.data) / Norm(f.vec.data)
        print(f"Relative residual: {rel_residual:e}")
        convergence = bool(rel_residual < rtol)
        return gfu, convergence

# --- Main function ---------------------------------
def build_spaces(mesh, contact_pairs, outer_contact_pairs=None, order_bulk=2, order_gb=1):
    """Construct the block mixed space holding bulk fields and GB/outer multipliers."""
    # Bulk displacement: no strong Dirichlet; corners handled by Nitsche
    V_Re = VectorH1(mesh, order=order_bulk)
    V_Im = VectorH1(mesh, order=order_bulk)

    gb_spaces = []
    name_order_trial = ["u_Re", "u_Im"]
    name_order_test  = ["v_Re", "v_Im"]

    # Grain-boundary multiplier spaces
    for (a, b), (_, right) in contact_pairs.items():
        bdry = mesh.Boundaries(f"core_{right}|slide_{right}")
        gb_spaces += [
            H1(mesh, order=order_gb, definedon=bdry),  # Re_s
            H1(mesh, order=order_gb, definedon=bdry),  # Im_s
            H1(mesh, order=order_gb, definedon=bdry),  # Re_n
            H1(mesh, order=order_gb, definedon=bdry),  # Im_n
        ]
        name_order_trial += [
            f"t_{a}_{b}_s_Re", f"t_{a}_{b}_s_Im",
            f"t_{a}_{b}_n_Re", f"t_{a}_{b}_n_Im"
        ]
        name_order_test += [
            f"r_{a}_{b}_s_Re", f"r_{a}_{b}_s_Im",
            f"r_{a}_{b}_n_Re", f"r_{a}_{b}_n_Im"
        ]
    print(f"Added {len(contact_pairs)*4} grain-boundary multiplier spaces.")

    outer_spaces = []

    combined_outer_pairs = outer_contact_pairs or {}

    for edge_key, data in combined_outer_pairs.items():
        plus_info = data["plus"]
        plus_prefix = plus_info["prefix"]
        core_expr = _core_boundary_expression(plus_info)
        boundary_expr = f"{core_expr}|slide_{plus_prefix}" if core_expr else f"slide_{plus_prefix}"
        bdry = mesh.Boundaries(boundary_expr)
        outer_spaces += [
            H1(mesh, order=order_gb, definedon=bdry),
            H1(mesh, order=order_gb, definedon=bdry),
            H1(mesh, order=order_gb, definedon=bdry),
            H1(mesh, order=order_gb, definedon=bdry),
        ]
        name_order_trial += [
            f"t_{edge_key}_s_Re", f"t_{edge_key}_s_Im",
            f"t_{edge_key}_n_Re", f"t_{edge_key}_n_Im",
        ]
        name_order_test += [
            f"r_{edge_key}_s_Re", f"r_{edge_key}_s_Im",
            f"r_{edge_key}_n_Re", f"r_{edge_key}_n_Im",
        ]
    if combined_outer_pairs:
        print(f"Added {len(combined_outer_pairs) * 4} outer boundary multiplier spaces.")
    else:
        print("No outer boundary multiplier spaces added.")

    fes = V_Re * V_Im
    with TaskManager():
        # build in groups to avoid too-large eval strings; join groups afterwards
        spaces = gb_spaces + outer_spaces
        n = len(spaces)
        group_fes_list = []
        if n:
            group_size = int(sqrt(n)) + 1
            grp_idx = 0
            for start in range(0, n, group_size):
                end = min(start + group_size, n)
                # build the product for this group explicitly (avoid eval)
                group_fes = spaces[start]
                for i in range(start + 1, end):
                    group_fes = group_fes * spaces[i]
                grp_idx += 1
                group_fes_list.append(group_fes)
                # optionally expose fes1, fes2, ... in globals for inspection
                try:
                    globals()[f"fes{grp_idx}"] = group_fes
                except Exception:
                    pass
                print(f"Constructed fes{grp_idx} from spaces {start+1}-{end} of {n}")

            # combine all group finite element spaces into the final fes
            for idx, gf in enumerate(group_fes_list, start=1):
                fes = fes * gf
            print(f"Combined {len(group_fes_list)} groups into fes")
        else:
            print("No additional multiplier spaces to add.")
    print("Constructed finite element space")

    trial_tuple = fes.TrialFunction()
    test_tuple  = fes.TestFunction()

    if len(trial_tuple) != len(name_order_trial):
        raise ValueError(f"Trial arity mismatch: {len(trial_tuple)} vs {len(name_order_trial)}")
    if len(test_tuple) != len(name_order_test):
        raise ValueError(f"Test arity mismatch: {len(test_tuple)} vs {len(name_order_test)}")

    sym_dict = {n: f for n, f in zip(name_order_trial, trial_tuple)}
    sym_dict.update({n: f for n, f in zip(name_order_test, test_tuple)})
    sym = SimpleNamespace(**sym_dict)
    print(f"Constructed finite element space with",fes.ndof," dofs.")

    # Build a name->index map so downstream code can grab multipliers safely
    trial_index = {name: idx for idx, name in enumerate(name_order_trial)}

    # Precompute tangential multiplier indices per contact pair
    gb_tangent_indices = {}
    for (a, b) in contact_pairs.keys():
        key = (a, b)
        gb_tangent_indices[key] = (
            trial_index[f"t_{a}_{b}_s_Re"],
            trial_index[f"t_{a}_{b}_s_Im"],
        )

    return (fes, V_Re, V_Im, sym, gb_tangent_indices)

def solve_rve(spaces, mesh, contact_pairs, outer_contact_pairs,
    gamma,
    nu=0.30,
    mu=1.0,
    omega=1.0,
    solver='direct',
    rtol=1e-4,
    corner_bnd=None):
    """Solve the EAGBS mixed problem for a given macro loading tensor."""
    Gamma, lam = _setup_material_properties(gamma, nu, mu)

    # Spaces
    fes, V_Re, V_Im, sym = spaces[0], spaces[1], spaces[2], spaces[3]

    # Bilinear & linear forms (elasticity + GB + periodic)
    a = _assemble_bilinear_form(
        mesh,
        sym,
        contact_pairs,
        outer_contact_pairs,
        lam,
        mu,
        omega,
        fes,
    )
    f = _assemble_linear_form(
        mesh,
        sym,
        Gamma,
        fes,
        outer_contact_pairs,
    )

    # Macro affine displacement
    disp = CoefficientFunction((x, y))
    CF_u = Gamma * disp

    # Nitsche corner enforcement on LB|LT|RB
    _add_corner_penalty(a, f, mesh, sym, CF_u, corner_bnd)

    # Solve (no special handling of corners needed now)
    gfu, convergence = _solve(a, f, fes, solver, rtol)

    return gfu, mesh, convergence

