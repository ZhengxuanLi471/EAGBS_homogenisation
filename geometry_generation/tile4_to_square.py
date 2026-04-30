#!/usr/bin/env python3
"""Build a non-periodic square domain from a 2D periodic Neper tessellation.

The script tiles a periodic input PLY across a small grid (default 2x2), recenters
it at the origin, and clips a target square (default area=1, i.e. side length 1).
This preserves polygon topology by clipping against the target square before
rebuilding vertices/faces and writing a new Neper-style PLY file.
"""
from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple

import numpy as np

EPS_MERGE = 1e-7
EPS_SNAP = 1e-8


def read_neper_tess_ply(filename: str) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header: List[str] = []
    i = 0
    n_vertices = n_faces = 0

    while True:
        line = lines[i]
        header.append(line)
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[2])
        elif line.startswith("element face"):
            n_faces = int(line.split()[2])
        elif line.startswith("end_header"):
            i += 1
            break
        i += 1

    verts = np.array([
        list(map(float, lines[i + j].split()[:3])) for j in range(n_vertices)
    ])

    face_start = i + n_vertices
    faces: List[List[int]] = []
    for j in range(n_faces):
        parts = lines[face_start + j].split()
        k = int(parts[0])
        faces.append(list(map(int, parts[1 : 1 + k])))

    return header, verts, faces


def clip_polygon_axis(poly: Sequence[Sequence[float]], axis: int, value: float, *, keep_greater: bool) -> List[List[float]]:
    if not poly:
        return []

    def coord(point: Sequence[float]) -> float:
        return point[axis]

    result: List[List[float]] = []
    n = len(poly)
    for idx in range(n):
        curr = poly[idx]
        prev = poly[idx - 1]

        curr_in = coord(curr) >= value if keep_greater else coord(curr) <= value
        prev_in = coord(prev) >= value if keep_greater else coord(prev) <= value

        if prev_in and curr_in:
            result.append([curr[0], curr[1]])
        elif prev_in and not curr_in:
            t = (value - coord(prev)) / (coord(curr) - coord(prev))
            inter = [
                prev[0] + t * (curr[0] - prev[0]),
                prev[1] + t * (curr[1] - prev[1]),
            ]
            result.append(inter)
        elif (not prev_in) and curr_in:
            t = (value - coord(prev)) / (coord(curr) - coord(prev))
            inter = [
                prev[0] + t * (curr[0] - prev[0]),
                prev[1] + t * (curr[1] - prev[1]),
            ]
            result.append(inter)
            result.append([curr[0], curr[1]])

    return result


def clip_polygon_box(poly: Sequence[Sequence[float]], x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]) -> List[List[float]]:
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    poly = clip_polygon_axis(poly, 0, xmin, keep_greater=True)
    poly = clip_polygon_axis(poly, 0, xmax, keep_greater=False)
    poly = clip_polygon_axis(poly, 1, ymin, keep_greater=True)
    poly = clip_polygon_axis(poly, 1, ymax, keep_greater=False)
    return poly


def build_tiled_polys(
    verts: np.ndarray,
    faces: Sequence[Sequence[int]],
    tile_size: float,
    square_side: float,
    grid: int,
) -> List[List[List[float]]]:
    if grid < 2:
        raise ValueError("grid must be at least 2 to cover the interior square")

    x_bounds = (-square_side / 2.0, square_side / 2.0)
    y_bounds = (-square_side / 2.0, square_side / 2.0)

    offsets: List[Tuple[float, float]] = []
    half = grid // 2
    for ix in range(grid):
        for iy in range(grid):
            dx = (ix - half) * tile_size
            dy = (iy - half) * tile_size
            offsets.append((dx, dy))

    polys: List[List[List[float]]] = []
    for face in faces:
        base = [[verts[idx, 0], verts[idx, 1]] for idx in face]
        for dx, dy in offsets:
            shifted = [[x + dx, y + dy] for x, y in base]
            clipped = clip_polygon_box(shifted, x_bounds, y_bounds)
            if len(clipped) >= 3:
                polys.append(clipped)

    return polys


def build_vertices_and_faces(
    polys: Sequence[Sequence[Sequence[float]]],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, List[List[int]]]:
    verts: List[List[float]] = []
    face_indices: List[List[int]] = []

    for poly in polys:
        start = len(verts)
        verts.extend([[x, y, 0.0] for x, y in poly])
        face_indices.append(list(range(start, start + len(poly))))

    if not verts:
        return np.zeros((0, 3)), []

    verts_np = np.array(verts)
    key = np.round(verts_np[:, :2] / EPS_MERGE).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)

    merged = np.zeros((inverse.max() + 1, 3))
    for idx, target in enumerate(inverse):
        merged[target] = verts_np[idx]

    faces_clean: List[List[int]] = []
    for face in face_indices:
        remapped = [inverse[idx] for idx in face]
        collapsed: List[int] = []
        for vid in remapped:
            if not collapsed or collapsed[-1] != vid:
                collapsed.append(vid)
        if len(collapsed) >= 3 and len(set(collapsed)) >= 3:
            faces_clean.append(collapsed)

    bounds = {0: x_bounds, 1: y_bounds}
    for vidx in range(merged.shape[0]):
        for axis in (0, 1):
            low, high = bounds[axis]
            val = merged[vidx, axis]
            if abs(val - low) <= EPS_SNAP:
                merged[vidx, axis] = low
            elif abs(val - high) <= EPS_SNAP:
                merged[vidx, axis] = high

    return merged, faces_clean


def write_neper_tess_ply(filename: str, header: Sequence[str], verts: np.ndarray, faces: Sequence[Sequence[int]]) -> None:
    header_out = list(header)
    n_vertices = verts.shape[0]
    n_faces = len(faces)

    for idx, line in enumerate(header_out):
        if line.startswith("element vertex"):
            header_out[idx] = f"element vertex {n_vertices}"
        elif line.startswith("element face"):
            header_out[idx] = f"element face {n_faces}"
        elif line.startswith("element cell"):
            header_out[idx] = f"element cell {n_faces}"
        elif line.startswith("end_header"):
            break

    with open(filename, "w", encoding="utf-8") as f:
        for line in header_out:
            f.write(line + "\n")

        for x, y, z in verts:
            f.write(f"{x:.12f} {y:.12f} {z:.12f}\n")

        for face in faces:
            f.write(str(len(face)) + " " + " ".join(map(str, face)) + "\n")

        for idx in range(n_faces):
            f.write(f"1 {idx}\n")


def make_center_square_from_periodic(
    input_ply: str,
    output_ply: str,
    tile_size: float,
    target_square_area: float,
    grid: int,
) -> None:
    if target_square_area <= 0:
        raise ValueError("target_square_area must be positive")

    square_side = math.sqrt(target_square_area)
    header, verts, faces = read_neper_tess_ply(input_ply)
    polys = build_tiled_polys(verts, faces, tile_size, square_side, grid)
    x_bounds = (-square_side / 2.0, square_side / 2.0)
    y_bounds = (-square_side / 2.0, square_side / 2.0)
    new_verts, new_faces = build_vertices_and_faces(polys, x_bounds, y_bounds)
    write_neper_tess_ply(output_ply, header, new_verts, new_faces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile a periodic Neper PLY (xy-periodic) into a square domain centered at the "
            "origin and clip a target square (default area = 1)."
        )
    )
    parser.add_argument("input", help="periodic input PLY")
    parser.add_argument("output", help="output PLY for the clipped square domain")
    parser.add_argument("--tile-size", type=float, default=1.0, help="side length of one periodic tile (default: 1.0)")
    parser.add_argument(
        "--square-area",
        type=float,
        default=1.0,
        help="area of the target centered square to keep (default: 1.0)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=2,
        help="number of tiles per axis to assemble before clipping (default: 2 => four tiles)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_center_square_from_periodic(
        input_ply=args.input,
        output_ply=args.output,
        tile_size=args.tile_size,
        target_square_area=args.square_area,
        grid=args.grid,
    )
    print("Done.")
