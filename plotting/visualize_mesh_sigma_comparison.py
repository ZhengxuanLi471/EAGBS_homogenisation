#!/usr/bin/env python3
# Side-by-side tessellation comparison: σ_a = 0.05 vs 0.5.

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
SMALL_SIGMA = (0.05, os.path.join(_HERE, "..", "geometry_generation", "sigma_0.05", "tessellation_output.json"))
LARGE_SIGMA = (0.5,  os.path.join(_HERE, "..", "geometry_generation", "sigma_0.5",  "tessellation_output.json"))

plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 26,
    'axes.titlesize': 28,
    'legend.fontsize': 18,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
})

color_palette = [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
    '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
    '#ccebc5', '#ffed6f'
]


def identify_periodic_grains(pts, regions, tol=1e-6):
    pts_array = np.array(pts)
    x_min, x_max = pts_array[:, 0].min(), pts_array[:, 0].max()
    y_min, y_max = pts_array[:, 1].min(), pts_array[:, 1].max()

    left_verts = {i for i, pt in enumerate(pts) if abs(pt[0] - x_min) < tol}
    right_verts = {i for i, pt in enumerate(pts) if abs(pt[0] - x_max) < tol}
    top_verts = {i for i, pt in enumerate(pts) if abs(pt[1] - y_max) < tol}
    bottom_verts = {i for i, pt in enumerate(pts) if abs(pt[1] - y_min) < tol}

    periodic_pairs = {}
    for lv in left_verts:
        ly = pts[lv][1]
        for rv in right_verts:
            if abs(ly - pts[rv][1]) < tol:
                periodic_pairs[lv] = rv
                periodic_pairs[rv] = lv
                break
    for bv in bottom_verts:
        bx = pts[bv][0]
        for tv in top_verts:
            if abs(bx - pts[tv][0]) < tol:
                periodic_pairs[bv] = tv
                periodic_pairs[tv] = bv
                break

    def get_grain_edges(region):
        edges = set()
        n = len(region)
        for i in range(n):
            v1 = region[i] - 1
            v2 = region[(i + 1) % n] - 1
            edges.add(tuple(sorted([v1, v2])))
        return edges

    grain_edges = {g: get_grain_edges(region) for g, region in enumerate(regions)}

    def get_boundary_edges(edges, boundary_verts):
        return {e for e in edges if e[0] in boundary_verts and e[1] in boundary_verts}

    left_edge_to_grain = {}
    right_edge_to_grain = {}
    top_edge_to_grain = {}
    bottom_edge_to_grain = {}

    for g, edges in grain_edges.items():
        for e in get_boundary_edges(edges, left_verts):
            left_edge_to_grain[e] = g
        for e in get_boundary_edges(edges, right_verts):
            right_edge_to_grain[e] = g
        for e in get_boundary_edges(edges, top_verts):
            top_edge_to_grain[e] = g
        for e in get_boundary_edges(edges, bottom_verts):
            bottom_edge_to_grain[e] = g

    parent = list(range(len(regions)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for left_edge, g1 in left_edge_to_grain.items():
        v1, v2 = left_edge
        pv1, pv2 = periodic_pairs.get(v1), periodic_pairs.get(v2)
        if pv1 is not None and pv2 is not None:
            partner_edge = tuple(sorted([pv1, pv2]))
            if partner_edge in right_edge_to_grain:
                union(g1, right_edge_to_grain[partner_edge])

    for bottom_edge, g1 in bottom_edge_to_grain.items():
        v1, v2 = bottom_edge
        pv1, pv2 = periodic_pairs.get(v1), periodic_pairs.get(v2)
        if pv1 is not None and pv2 is not None:
            partner_edge = tuple(sorted([pv1, pv2]))
            if partner_edge in top_edge_to_grain:
                union(g1, top_edge_to_grain[partner_edge])

    corners = (left_verts | right_verts) & (top_verts | bottom_verts)
    corner_grains = set()
    for g_idx, region in enumerate(regions):
        region_verts = set(v - 1 for v in region)
        if region_verts & corners:
            corner_grains.add(g_idx)
    corner_grains_list = list(corner_grains)
    for i in range(1, len(corner_grains_list)):
        union(corner_grains_list[0], corner_grains_list[i])

    grain_groups = {}
    grain_to_group = {}
    for g in range(len(regions)):
        rep = find(g)
        grain_to_group[g] = rep
        if rep not in grain_groups:
            grain_groups[rep] = []
        grain_groups[rep].append(g)

    return grain_groups, grain_to_group


def build_adjacency_graph(regions, grain_to_group):
    edge_to_grains = {}
    for g_idx, region in enumerate(regions):
        n = len(region)
        for i in range(n):
            edge = tuple(sorted([region[i], region[(i + 1) % n]]))
            edge_to_grains.setdefault(edge, []).append(g_idx)

    adjacency = {rep: set() for rep in set(grain_to_group.values())}
    for edge, grains in edge_to_grains.items():
        if len(grains) == 2:
            rep1 = grain_to_group[grains[0]]
            rep2 = grain_to_group[grains[1]]
            if rep1 != rep2:
                adjacency[rep1].add(rep2)
                adjacency[rep2].add(rep1)
    return adjacency


def graph_coloring(adjacency):
    colors = {}
    for node in sorted(adjacency.keys()):
        neighbor_colors = {colors[n] for n in adjacency[node] if n in colors}
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[node] = c
    return colors


def load_tessellation(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    first_key = sorted(data.keys())[0]
    pts, regions = data[first_key]
    return pts, regions, first_key


def plot_tessellation(ax, pts, regions, panel_label):
    grain_groups, grain_to_group = identify_periodic_grains(pts, regions)
    adjacency = build_adjacency_graph(regions, grain_to_group)
    colors_map = graph_coloring(adjacency)

    for region_idx, region in enumerate(regions):
        poly_pts = [pts[i - 1] for i in region]
        poly = Polygon(poly_pts, closed=True)
        rep = grain_to_group[region_idx]
        poly.set_facecolor(color_palette[colors_map[rep] % len(color_palette)])
        poly.set_alpha(0.7)
        poly.set_edgecolor('none')
        ax.add_patch(poly)

    pts_array = np.array(pts)

    edge_count = {}
    for region in regions:
        n = len(region)
        for k in range(n):
            edge = tuple(sorted([region[k], region[(k + 1) % n]]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    gb_lines = []
    ext_lines = []
    for edge, count in edge_count.items():
        p0, p1 = pts[edge[0] - 1], pts[edge[1] - 1]
        if count == 2:
            gb_lines.append([p0, p1])
        elif count == 1:
            ext_lines.append([p0, p1])

    ax.add_collection(LineCollection(gb_lines, colors='#2c3e50', linewidths=1.5))
    ax.add_collection(LineCollection(ext_lines, colors='#e74c3c', linewidths=2.0))

    ax.set_xlim(pts_array[:, 0].min() - 0.01, pts_array[:, 0].max() + 0.01)
    ax.set_ylim(pts_array[:, 1].min() - 0.01, pts_array[:, 1].max() + 0.01)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(-0.06, 0.98, panel_label, transform=ax.transAxes,
            fontsize=32, fontweight='bold', va='top', ha='left')


# ── main ────────────────────────────────────────────────────────────────────
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 9))

for ax, (sigma_val, path), panel_label in [
    (ax0, SMALL_SIGMA, "a)"),
    (ax1, LARGE_SIGMA, "b)"),
]:
    pts, regions, key = load_tessellation(path)
    print(f"σ={sigma_val}: key={key}, {len(regions)} grains, {len(pts)} vertices")
    plot_tessellation(ax, pts, regions, panel_label)

fig.tight_layout(pad=2)
out_path = "mesh_tessellation_sigma_comparison.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
