import scipy.sparse as sp
import igl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist  
import numpy as np  
# import torch  
import triangle 
from escher.geometry.sanity_checks import check_triangle_orientation
import gpytoolbox
import enum

class RemeshMethod(enum.Enum):
    TRIANGLE = 1
    GPYTOOLBOX = 2
    SimpleLaplacianSmooth = 3

def compute_comprehensive_distortion(vertices, faces, reference_vertices=None):  
    """  
    Compute comprehensive mesh quality metrics for distortion detection  
      
    Args:  
        vertices: Current vertex positions (N x 2)  
        faces: Triangle faces (M x 3)  
        reference_vertices: Original vertex positions for Jacobian computation  
      
    Returns:  
        distortion_score: Combined distortion score (0-1, higher = more distorted)  
    """  
    distortion_score = 0.0  
      
    # 1. Triangle Aspect Ratio  
    aspect_ratios = []  
    for face in faces:  
        triangle_verts = vertices[face]  
        edges = pdist(triangle_verts)  # Compute all pairwise distances  
        aspect_ratio = np.max(edges) / np.min(edges)  
        aspect_ratios.append(aspect_ratio)  
      
    aspect_ratios = np.array(aspect_ratios)  
    bad_aspect_ratio = np.mean(aspect_ratios > 3.0)  
    distortion_score += bad_aspect_ratio * 0.2  
      
    # 2. Triangle Area Variation  
    areas = []  
    for face in faces:  
        triangle_verts = vertices[face]  
        # Compute area using cross product  
        v1 = triangle_verts[1] - triangle_verts[0]  
        v2 = triangle_verts[2] - triangle_verts[0]  
        area = 0.5 * abs(np.cross(v1, v2))  
        areas.append(area)  
      
    areas = np.array(areas)  
    if len(areas) > 1:  
        area_cv = np.std(areas) / np.mean(areas)  # Coefficient of variation  
        distortion_score += min(area_cv, 1.0) * 0.15  
      
    # 3. Angle Quality  
    min_angles = []  
    max_angles = []  
    for face in faces:  
        triangle_verts = vertices[face]  
        angles = []  
        for i in range(3):  
            v1 = triangle_verts[(i+1)%3] - triangle_verts[i]  
            v2 = triangle_verts[(i+2)%3] - triangle_verts[i]  
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  
            cos_angle = np.clip(cos_angle, -1, 1)  
            angle = np.arccos(cos_angle) * 180 / np.pi  
            angles.append(angle)  
        min_angles.append(min(angles))  
        max_angles.append(max(angles))  
      
    min_angles = np.array(min_angles)  
    max_angles = np.array(max_angles)  
    bad_angles = np.sum((min_angles < 15) | (max_angles > 120)) / len(faces)  
    distortion_score += bad_angles * 0.2  
      
    # 4. Edge Length Ratio  
    edge_ratios = []  
    for face in faces:  
        triangle_verts = vertices[face]  
        edges = pdist(triangle_verts)  
        if len(edges) == 3:  # Should always be 3 for triangles  
            edge_ratio = np.max(edges) / np.min(edges)  
            edge_ratios.append(edge_ratio)  
      
    edge_ratios = np.array(edge_ratios)  
    bad_edge_ratios = np.mean((edge_ratios > 2.0) | (edge_ratios < 0.5))  
    distortion_score += bad_edge_ratios * 0.15  
      
    # # 5. Jacobian Determinant (requires reference vertices)  
    # if reference_vertices is not None:  
    #     jacobian_dets = []  
    #     for face in faces:  
    #         current_tri = vertices[face]  
    #         ref_tri = reference_vertices[face]  
              
    #         # Compute edge vectors  
    #         current_edges = current_tri[1:] - current_tri[0]  
    #         ref_edges = ref_tri[1:] - ref_tri[0]  
              
    #         # Compute Jacobian matrix (2x2 for 2D)  
    #         try:  
    #             J = np.linalg.solve(ref_edges.T, current_edges.T)  
    #             det_J = np.linalg.det(J)  
    #             jacobian_dets.append(abs(det_J))  
    #         except np.linalg.LinAlgError:  
    #             jacobian_dets.append(0.0)  # Degenerate case  
          
    #     jacobian_dets = np.array(jacobian_dets)  
    #     bad_jacobians = np.mean((jacobian_dets < 0.1) | (jacobian_dets > 5.0))  
    #     distortion_score += bad_jacobians * 0.2  
      
    # 6. Gaussian Curvature Variation  
    curvatures = []  
    for i in range(len(vertices)):  
        # Find faces containing vertex i  
        vertex_faces = [f for f in range(len(faces)) if i in faces[f]]  
        if len(vertex_faces) > 0:  
            angle_sum = 0.0  
            for face_idx in vertex_faces:  
                face = faces[face_idx]  
                vertex_pos = np.where(face == i)[0][0]  
                  
                # Get adjacent vertices  
                v1_idx = face[(vertex_pos + 1) % 3]  
                v2_idx = face[(vertex_pos + 2) % 3]  
                  
                # Compute angle at vertex i  
                v1 = vertices[v1_idx] - vertices[i]  
                v2 = vertices[v2_idx] - vertices[i]  
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  
                cos_angle = np.clip(cos_angle, -1, 1)  
                angle = np.arccos(cos_angle)  
                angle_sum += angle  
              
            # Discrete Gaussian curvature (angle defect)  
            curvature = 2 * np.pi - angle_sum  
            curvatures.append(abs(curvature))  
      
    if len(curvatures) > 1:  
        curvature_variation = np.std(curvatures)  
        distortion_score += min(curvature_variation, 1.0) * 0.1  
      
    return min(distortion_score, 1.0)  # Cap at 1.0





def split_boundary_by_discrete_2d_curvature(V, F):
    """
    Split the boundary of a 2D triangle mesh into 4 labeled sides
    ('top', 'bottom', 'left', 'right') and return only the start and end
    indices (into the boundary loop) for each side.

    Parameters
    ----------
    V : (n, 2) ndarray
        2D vertex coordinates
    F : (m, 3) ndarray
        Triangle face indices

    Returns
    -------
    sides_dict : dict
        Dictionary mapping 'top', 'bottom', 'left', 'right' to a tuple
        (start_idx, end_idx), where each is an index into the boundary array
    boundary : (k,) ndarray
        The ordered boundary vertex indices (into V)
    """

    # Step 1: Get ordered boundary loop
    boundary = igl.boundary_loop(F)
    if len(boundary)!=196:
        raise RuntimeError(f"Warning: Boundary length is {len(boundary)}, expected 196. This may affect segmentation.")
    loop = V[boundary]

    wrapped=[0,49,98,147,196]  # Assuming these are the indices of the corners in the boundary loop

    # Step 4: Split into segments and compute centroids
    segments = []
    # segment_indices = []
    seg_vertices = []
    for i in range(4):
        start = wrapped[i] % len(loop)
        end = wrapped[i + 1] % len(loop)
        if start < end:
            seg = boundary[start:end+1]
            seg_v = loop[start:end+1]
        else:
            seg = np.concatenate([boundary[start:], boundary[:end+1]])
            seg_v = np.vstack([loop[start:], loop[:end]])
        segments.append(seg)
        seg_vertices.append(seg_v)
        # segment_indices.append(seg_indices)

    centroids = np.array([seg.mean(axis=0) for seg in seg_vertices])

    # Step 5: Assign labels based on centroid location
    top_idx    = np.argmax(centroids[:, 1])
    bottom_idx = np.argmin(centroids[:, 1])
    left_idx   = np.argmin(centroids[:, 0])
    right_idx  = np.argmax(centroids[:, 0])

    # Step 6: Ensure unique labels
    assigned = {}
    used = set()
    for label, idx in [('top', top_idx), ('bottom', bottom_idx),
                       ('left', left_idx), ('right', right_idx)]:
        if idx not in used:
            assigned[label] = idx
            used.add(idx)
        else:
            for alt in range(4):
                if alt not in used:
                    assigned[label] = alt
                    used.add(alt)
                    break

    # Step 7: Build sides_dict using only boundary index pairs
    sides_dict = {}
    for label, seg_idx in assigned.items():
        # start, end = segment_indices[seg_idx]
        sides_dict[label] = segments[seg_idx]

    return sides_dict


def get_max_edge_length(V, F):
    # Extract the vertex coordinates of each triangle
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    
    # Compute lengths of all edges
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    
    # Return the maximum
    return np.max([e0.max(), e1.max(), e2.max()])

def plot_sides(sides,vertices):
    import matplotlib.pyplot as plt
    for i, side in sides.items():
        side_vertices = vertices[side]
        plt.plot(side_vertices[:, 0], side_vertices[:, 1], label=f"Side {i}")
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title("4 Split Sides")
    plt.show()

def compute_average_area(V, F):
    areas = []
    for tri in F:
        v0, v1, v2 = V[tri]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
    return np.max(areas)



def triangle_orientation_2d(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # Signed area * 2
    signed_area2 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
                   (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])

    return np.sign(signed_area2)  # +1 CCW, -1 CW, 0 degenerate



# def repair_mesh(V, F, area_eps=1e-6, merge_eps=1e-12, keep_largest=True):
#     # -------------------
#     # 1. Remove degenerate faces
#     v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
#     areas = 0.5 *np.cross(v1 - v0, v2 - v0)
#     if areas.max()<0:
#         areas=-areas
#     mask = areas > area_eps
#     F = F[mask]

#     # -------------------
#     # 2. Merge duplicate vertices
#     # V,F,*rest= igl.remove_duplicates(V,F, 1e-7)

#     # -------------------
#     # 3. Remove unreferenced vertices
#     V, F, *rest = igl.remove_unreferenced(V, F)

#     # -------------------
#     # 4. Remove small disconnected components
#     # if keep_largest:
#     #     C = igl.connected_components(F)[0]
#     #     largest_comp = np.argmax(np.bincount(C))
#     #     mask = C == largest_comp
#     #     F = F[mask]
#     #     V, F, *rest = igl.remove_unreferenced(V, F)

#     # -------------------
#     # 5. Fix orientation & normals (trimesh)
#     # mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
#     # mesh.remove_degenerate_faces()
#     # mesh.remove_duplicate_faces()
#     # mesh.remove_infinite_values()
#     # mesh.remove_unreferenced_vertices()

#     # # Make normals consistent & outward
#     # mesh.fix_normals()

#     # return np.array(mesh.vertices), np.array(mesh.faces)
#     return V, F


def laplacian_smooth_manual(V, F, iterations=10, lam=0.5, fixed_boundary=True):
    """
    Laplacian smoothing using igl.adjacency_list for neighbor finding.
    """
    V_smooth = V.copy()
    n_vertices = V.shape[0]

    # Get adjacency list from libigl
    adjacency = igl.adjacency_list(F)  # list of lists

    # Detect boundary vertices if needed
    boundary_mask = np.zeros(n_vertices, dtype=bool)
    if fixed_boundary:
        boundary_vertices = igl.boundary_loop(F)
        boundary_mask[boundary_vertices] = True

    # Iterative smoothing
    for _ in range(iterations):
        new_positions = V_smooth.copy()
        for i in range(n_vertices):
            if fixed_boundary and boundary_mask[i]:
                continue
            if len(adjacency[i]) == 0:
                continue
            avg_neighbor = np.mean(V_smooth[adjacency[i]], axis=0)
            new_positions[i] = (1 - lam) * V_smooth[i] + lam * avg_neighbor
        V_smooth = new_positions

    return V_smooth



def barycentric_remesh_optimization(vertices,faces,boundary_indices,remesh_method:RemeshMethod=RemeshMethod.TRIANGLE, use_triangle=True, use_gpytoolbox=False):
    """
    Complete remesh optimization using barycentric weight recomputation  
      
    Args:  
        vertices: Current vertex positions  
        faces: Current triangle faces  
        constraint_data: Current constraint system  
        distortion_threshold: Threshold for triggering remesh  
      
    Returns:  
        new_vertices: Remeshed vertices  
        new_faces: New triangle connectivity  
        new_edge_pairs: New edge connectivity  
        new_weights: Barycentric weights  
        new_constraint_data: Rebuilt constraint system  
    """  
    #get triangle areas
    # max_area = compute_average_area(vertices, faces)
    # print(f"Triangle area - Max: {max_area}")  
    
    if remesh_method == RemeshMethod.TRIANGLE:
        # Convert to boundary segments (pairs of vertex indices)
        segments = np.column_stack([boundary_indices, np.roll(boundary_indices, -1)])

        # Step 4: Build triangle input
        A = {
            'vertices': vertices,
            'segments': segments,
        }

        # Step 5: Call triangle to triangulate the polygon
        # B = triangle.triangulate(A, f'pqY')  # 'p' = PSLG (respect segments)
        B = triangle.triangulate(A, f'p')  # 'p' = PSLG (respect segments)
        # Extract vertices and faces
        new_vertices = B['vertices']
        new_faces = B['triangles']
    elif remesh_method == RemeshMethod.GPYTOOLBOX:
        vertices_3d=np.hstack([vertices, np.zeros((vertices.shape[0], 1))])

        #get max edge length from surface
        max_edge_length = get_max_edge_length(vertices, faces) 

        new_vertices_3d, new_faces = gpytoolbox.remesh_botsch(vertices_3d, faces, 20, max_edge_length, False)
        new_vertices=new_vertices_3d[:,:2]
        new_vertices=new_vertices.astype(np.float32)
        new_faces=new_faces.astype(np.int64)
        new_vertices=np.ascontiguousarray(new_vertices)
        new_faces=np.ascontiguousarray(new_faces)
    elif remesh_method == RemeshMethod.SimpleLaplacianSmooth:
        new_vertices = laplacian_smooth_manual(vertices, faces, lam=0.5, iterations=10)
        new_faces = faces

    # print(new_vertices.flags['C_CONTIGUOUS'])  # True
    # print(new_vertices.flags['F_CONTIGUOUS'])  # False
    # print(new_faces.flags['C_CONTIGUOUS'])  # True
    # print(new_faces.flags['F_CONTIGUOUS'])  # False

    # new_vertices,new_faces=repair_mesh(new_vertices, new_faces)

    view_mesh = True
    if view_mesh:
        old_mesh= {
            'vertices': vertices,
            'triangles': faces
        }

        B={
            'vertices': new_vertices,
            'triangles': new_faces
        }
        triangle.compare(plt, old_mesh, B)
        plt.show()

    # # check_triangle_orientation(new_vertices, new_faces)
    # orientation = triangle_orientation_2d(new_vertices, new_faces)


    L = igl.cotmatrix(new_vertices, new_faces)

    # Extract edges and weights from sparse matrix
    L_coo = L.tocoo()
    new_weights = []

    for i, j, w in zip(L_coo.row, L_coo.col, L_coo.data):
        if i < j:  # avoid duplicates and skip diagonals
            new_weights.append( -w)  # negate to get positive weight
      
    return new_vertices, new_faces, new_weights 
  