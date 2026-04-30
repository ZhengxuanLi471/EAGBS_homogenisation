#!/usr/bin/env python3
import math
import os

# lognormal shape parameters
siglist = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]

# convert lognormal sigma -> coefficient of variation
def sigin(sigma):
    return math.sqrt(math.exp(sigma**2) - 1)

n_grains = 50
n_tess = 100

for sig in siglist:
    sig_in = sigin(sig)
    dirname = f"sigma_{sig}"

    # ---------------------------
    # Create directory
    # ---------------------------
    print(f"mkdir -p {dirname}")
    print(f"cd {dirname}")

    # Copy helper scripts
    print("cp ../tile4_to_square.py .")
    print("cp ../read.py .")

    # ---------------------------
    # Generate tessellations
    # ---------------------------
    print(f"echo '=== [{dirname}] Starting tessellation generation ==='")
    for seed in range(1, n_tess + 1):
        print(f"echo '[{dirname}] Generating tessellation seed {seed}/{n_tess}'")
        parts = [
            "neper -T",
            "-dim 2",
            f"-id {seed}",
            '-domain "square(1,1)"',
            "-periodicity 1",
            f"-n {n_grains}",
            (
                "-morpho "
                f"\"diameq:lognormal(1,{sig_in:.5f}),"
                "1-circularity:lognormal(0.100,0.03)\""
            ),
            "-format tess,ply",
            f"-o seed_{seed}",
        ]
        print(" \\\n  ".join(parts) + " > /dev/null 2>&1")

    # ---------------------------
    # Visualisation
    # ---------------------------
    print(f"echo '=== [{dirname}] Starting visualisation ==='")
    for seed in range(1, n_tess + 1):
        print(f"echo '[{dirname}] Visualising seed {seed}/{n_tess}'")
        print(
            " \\\n  ".join([
                "neper -V",
                f"seed_{seed}.tess",
                f"-print seed_{seed}",
            ]) + " > /dev/null 2>&1"
        )

    # ---------------------------
    # Post-process to square tile
    # ---------------------------
    print(f"echo '=== [{dirname}] Starting post-processing ==='")
    for seed in range(1, n_tess + 1):
        print(f"echo '[{dirname}] Post-processing seed {seed}/{n_tess}'")
        print(
            " \\\n  ".join([
                "python tile4_to_square.py",
                f"seed_{seed}.ply",
                f"seed_{seed}_square.ply",
            ])
        )

    print(f"echo '=== [{dirname}] DONE ==='")
    print("cd ..\n")
