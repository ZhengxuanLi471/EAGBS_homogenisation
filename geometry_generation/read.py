import numpy as np
import json

def read_ply_tessellation(filename):
    """
    Reads a 2D Neper tessellation in PLY format and produces:
        pts     = list of (x,y)
        regions = list of tuples of vertex indices (1-based), CCW order
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

    # ---------------------------
    # 1. Parse header
    # ---------------------------
    n_vertices = 0
    n_faces = 0
    n_cells = 0
    i = 0
    while not lines[i].startswith("end_header"):
        if lines[i].startswith("element vertex"):
            n_vertices = int(lines[i].split()[2])
        elif lines[i].startswith("element face"):
            n_faces = int(lines[i].split()[2])
        elif lines[i].startswith("element cell"):
            n_cells = int(lines[i].split()[2])
        i += 1
    header_end = i + 1

    # ---------------------------
    # 2. Read vertices
    # ---------------------------
    pts_all = []
    for j in range(n_vertices):
        x, y, z = map(float, lines[header_end + j].split())
        pts_all.append((x, y))  # ignore z (always 0)

    # ---------------------------
    # 3. Read faces (list of vertex indices)
    # ---------------------------
    faces_start = header_end + n_vertices
    faces = []
    for j in range(n_faces):
        parts = list(map(int, lines[faces_start + j].split()))
        k = parts[0]          # number of vertices in this face
        verts = parts[1:1+k]  # 0-based indices
        faces.append(verts)

    # ---------------------------
    # 4. Read cells (list of face indices)
    # ---------------------------
    cells_start = faces_start + n_faces
    cells = []
    for j in range(n_cells):
        parts = list(map(int, lines[cells_start + j].split()))
        k = parts[0]
        face_ids = parts[1:1+k]
        cells.append(face_ids)

    # ---------------------------
    # 5. For each cell, build its outer polygon
    # ---------------------------
    regions = []

    for face_ids in cells:
        # Collect all vertices from the faces belonging to this cell
        verts = []
        for fid in face_ids:
            verts.extend(faces[fid])  # fid is already 0-based
        # Remove duplicates while preserving order
        unique = []
        for v in verts:
            if v not in unique:
                unique.append(v)

        # Compute CCW sorting
        cell_pts = np.array([pts_all[v] for v in unique])
        cx, cy = np.mean(cell_pts[:,0]), np.mean(cell_pts[:,1])
        angles = np.arctan2(cell_pts[:,1] - cy, cell_pts[:,0] - cx)
        order = np.argsort(angles)
        ordered_vertices = [unique[k] + 1 for k in order]  # convert to 1-based

        regions.append(tuple(ordered_vertices))

    return pts_all, regions


# ---------------------------
# Example usage:
# ---------------------------

data = {}
for seed_idx in range(1, 220):
    fname = f"seed_{seed_idx}_square.ply"
    try:
        pts, regions = read_ply_tessellation(fname)
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue
    data[f"seeds_{seed_idx}"] = [pts, regions]
    print(f"Processed {fname}: points={len(pts)}, regions={len(regions)}")

print(len(data), "tessellations processed.")

# save output into a JSON file
with open("tessellation_output.json", "w") as out:
    json.dump(data, out)

with open("tessellation_output.json", "r") as f:
    data = json.load(f)

for seed in sorted(data.keys()):
    pts, regions = data[seed]
    print(seed)
    print("  first 5 points:")
    for p in pts[:5]:
        print("   ", p)
    print("  first 5 regions:")
    for r in regions[:5]:
        print("   ", r)
    print()
#save output into a file"""

'''file = 'try.ply'
pts, regions = read_ply_tessellation(file)
print(f"Processed {file}: points={len(pts)}, regions={len(regions)}")
print('regions=', regions)
print('pts=', pts)'''