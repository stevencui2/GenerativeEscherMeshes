
# import triangle
# import triangle.plot as tplot
# import matplotlib.pyplot as plt

# # Define a square polygon as a list of points (counter-clockwise)
# A = dict(vertices=[
#     [0, 0],   # v0
#     [1, 0],   # v1
#     [1, 1],   # v2
#     [0, 1]    # v3
# ],
# segments=[
#     [0, 1],
#     [1, 2],
#     [2, 3],
#     [3, 0]
# ])

# # Triangulate using default options
# B = triangle.triangulate(A, 'pa0.2')  # 'p' = PSLG (piecewise linear)

# # Create a figure and axes
# fig, ax = plt.subplots()
# ax.set_aspect('equal')  # ✅ correct way to set aspect

# # Plot the triangulation into your axes
# tplot(ax, **B)

# plt.title("2D Triangulated Mesh")
# plt.show()

import numpy as np
import igl
import triangle
import triangle.plot as tplot
import matplotlib.pyplot as plt



import numpy as np

def compute_average_area(V, F):
    areas = []
    for tri in F:
        v0, v1, v2 = V[tri]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
    return np.mean(areas), np.median(areas), np.min(areas), np.max(areas)


def get_edge_pairs( faces_npy):
    """Generate nx2 list of edge pairs (i,j) from faces_npy"""
    adjacency_list = igl.adjacency_list(faces_npy)
    edge_pairs = []
    for r, i in zip(adjacency_list, range(len(adjacency_list))):
        for j in r:
            if i < j:
                edge_pairs.append((i, j))
    edge_pairs = np.asarray(edge_pairs)
    return edge_pairs

# Step 1: Define 2D mesh (in XY plane)
# 3 triangles forming a square with a diagonal
V = np.array([
    [0.0, 0.0],  # 0
    [1.0, 0.0],  # 1
    [1.0, 1.0],  # 2
    [0.0, 1.0],  # 3
    [0.5, 0.5],  # 4 (center point)
])

# Two triangles for square: [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
F = np.array([
    [0, 1, 4],
    [1, 2, 4],
    [2, 3, 4],
    [3, 0, 4],
])

# Example usage:
mean_area, median_area, min_area, max_area = compute_average_area(V, F)
print(f"Average: {mean_area}, Median: {median_area}")
# Step 2: Use igl.boundary_loop to get ordered boundary vertex indices
loop = igl.boundary_loop(F)
print("Boundary loop indices:", loop)

# Step 3: Extract 2D boundary vertices and create segments
boundary_vertices = V[loop]  # shape (n, 2)
segments = [[i, (i + 1) % len(loop)] for i in range(len(loop))]

# Step 4: Build triangle input
A = {
    'vertices': boundary_vertices,
    'segments': segments,
}

# Step 5: Call triangle to triangulate the polygon
B = triangle.triangulate(A, 'pqa0.5D')  # 'p' = PSLG (respect segments)

# Extract vertices and faces
vertices = B['vertices']
faces = B['triangles']

unique_edges = get_edge_pairs( faces)
# Compute cotangent Laplacian matrix using libigl
L = igl.cotmatrix(vertices, faces)

# Extract edges and weights from sparse matrix
L_coo = L.tocoo()
# edge_pairs = np.column_stack((L_coo.row, L_coo.col))
# weights = -L_coo.data  # libigl returns negative cotangent weights

# # Filter duplicate edges (since matrix is symmetric)
# edge_pairs = np.sort(edge_pairs, axis=1)
# unique_edges, unique_idx = np.unique(edge_pairs, axis=0, return_index=True)
# unique_weights = weights[unique_idx]

# L_coo = L.tocoo()
edge_weights = []

for i, j, w in zip(L_coo.row, L_coo.col, L_coo.data):
    if i < j:  # avoid duplicates and skip diagonals
        edge_weights.append(((i, j), -w))  # negate to get positive weight


# Step 6: Visualize
fig, ax = plt.subplots()
ax.set_aspect('equal')
tplot(ax, **B)
plt.title("2D Re-triangulated Boundary (via triangle)")
plt.show()

