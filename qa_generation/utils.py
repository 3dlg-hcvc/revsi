import os
import csv
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
from scipy.spatial import cKDTree
from collections.abc import Hashable, Iterable, Mapping


def write_csv(output_path: str, rows: Iterable[Mapping[str, object]], fieldnames: Iterable[str]) -> None:
    """Write rows to a CSV file, creating the parent directory when needed."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def sample_obb_surface_points(o3d_mesh: o3d.geometry.TriangleMesh, poisson_radius: float = 0.01) -> np.ndarray:
    """Sample points on an oriented bounding-box mesh surface with Poisson-disk sampling."""
    mesh_faces = np.asarray(o3d_mesh.triangles)
    mesh_vertices = np.asarray(o3d_mesh.vertices)
    face_indices, barycentric_coords = pcu.sample_mesh_poisson_disk(mesh_vertices, mesh_faces, num_samples=-1, radius=poisson_radius)
    sampled_points = pcu.interpolate_barycentric_coords(mesh_faces, face_indices, barycentric_coords, mesh_vertices)
    return sampled_points.astype(np.float32)


def calculate_nearest_distance(query_points: np.ndarray, reference_points: np.ndarray) -> float:
    """Return the minimum nearest-neighbor distance from query_points to reference_points."""
    reference_tree = cKDTree(reference_points)
    nearest_dists, _ = reference_tree.query(query_points, k=1)
    return nearest_dists.min()


def nearest_distance_to_tree(query_points: np.ndarray, reference_tree: cKDTree) -> float:
    """Return the minimum nearest-neighbor distance from points to a prebuilt KD-tree."""
    nearest_dists, _ = reference_tree.query(query_points, k=1)
    return float(nearest_dists.min())


def are_3d_obbs_overlapping(obb1: o3d.geometry.OrientedBoundingBox, obb2: o3d.geometry.OrientedBoundingBox) -> bool:
    """
    Return whether two 3D oriented bounding boxes intersect or touch.

    Uses the Separating Axis Theorem (SAT) over the 15 3D OBB candidate axes.
    """

    center_1 = np.asarray(obb1.center, dtype=np.float32)
    center_2 = np.asarray(obb2.center, dtype=np.float32)

    rotation_1 = np.asarray(obb1.R, dtype=np.float32)
    rotation_2 = np.asarray(obb2.R, dtype=np.float32)

    half_extent_1 = 0.5 * np.asarray(obb1.extent, dtype=np.float32)
    half_extent_2 = 0.5 * np.asarray(obb2.extent, dtype=np.float32)

    relative_rotation = rotation_1.T @ rotation_2
    center_delta = rotation_1.T @ (center_2 - center_1)
    abs_relative_rotation = np.abs(relative_rotation) + 1e-6

    for i in range(3):
        radius_1 = half_extent_1[i]
        radius_2 = half_extent_2[0] * abs_relative_rotation[i,0] + half_extent_2[1] * abs_relative_rotation[i,1] + half_extent_2[2] * abs_relative_rotation[i,2]
        if abs(center_delta[i]) > radius_1 + radius_2:
            return False

    for j in range(3):
        radius_1 = half_extent_1[0]*abs_relative_rotation[0,j] + half_extent_1[1]*abs_relative_rotation[1,j] + half_extent_1[2]*abs_relative_rotation[2,j]
        radius_2 = half_extent_2[j]
        projected_delta = abs(center_delta[0] * relative_rotation[0,j] + center_delta[1] * relative_rotation[1,j] + center_delta[2] * relative_rotation[2,j])
        if projected_delta > radius_1 + radius_2:
            return False

    for i in range(3):
        for j in range(3):
            radius_1 = half_extent_1[(i+1) % 3] * abs_relative_rotation[(i+2) % 3, j] + half_extent_1[(i+2) % 3] * abs_relative_rotation[(i+1) % 3, j]
            radius_2 = half_extent_2[(j+1) % 3] * abs_relative_rotation[i, (j+2) % 3] + half_extent_2[(j+2) % 3] * abs_relative_rotation[i, (j+1) % 3]
            projected_delta = abs(center_delta[(i+2) % 3] * relative_rotation[(i+1) % 3, j] - center_delta[(i+1) % 3] * relative_rotation[(i+2) % 3, j])
            if projected_delta > radius_1 + radius_2:
                return False

    return True


def _obb3d_to_obb2d_xy(obb: o3d.geometry.OrientedBoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a 3D OBB to its XY-plane 2D center, rotation axes, and half extent."""
    projected_center = np.asarray(obb.center)[:2]
    rotation_3d = np.asarray(obb.R)
    projected_axis_0 = rotation_3d[:2, 0]

    axis_norm = np.linalg.norm(projected_axis_0)
    if axis_norm > 1e-6:
        unit_axis_0 = projected_axis_0 / axis_norm
    else:
        unit_axis_0 = np.array([1.0, 0.0])
    unit_axis_1 = np.array([-unit_axis_0[1], unit_axis_0[0]])

    rotation_2d = np.stack([unit_axis_0, unit_axis_1], axis=1)
    half_extent = 0.5 * np.asarray(obb.extent)[:2]
    return projected_center, rotation_2d, half_extent


def are_xy_projections_overlapping(obb1: o3d.geometry.OrientedBoundingBox, obb2: o3d.geometry.OrientedBoundingBox) -> bool:
    """Return whether two 3D OBBs overlap after projection onto the XY plane."""
    center_1, rotation_1, half_extent_1 = _obb3d_to_obb2d_xy(obb1)
    center_2, rotation_2, half_extent_2 = _obb3d_to_obb2d_xy(obb2)
    relative_rotation = rotation_1.T @ rotation_2
    center_delta = rotation_1.T @ (center_2 - center_1)
    abs_relative_rotation = np.abs(relative_rotation) + 1e-6

    for i in range(2):
        radius_1 = half_extent_1[i]
        radius_2 = half_extent_2[0] * abs_relative_rotation[i,0] + half_extent_2[1] * abs_relative_rotation[i,1]
        if abs(center_delta[i]) > radius_1 + radius_2:
            return False

    for j in range(2):
        radius_1 = half_extent_1[0] * abs_relative_rotation[0,j] + half_extent_1[1] * abs_relative_rotation[1,j]
        radius_2 = half_extent_2[j]
        projected_delta = abs(center_delta[0] * relative_rotation[0,j] + center_delta[1] * relative_rotation[1,j])
        if projected_delta > radius_1 + radius_2:
            return False
    return True


def find_valid_4_tuples_from_pair_scores(
    items: Iterable[Hashable],
    pair_score: Mapping[object, float],
    threshold: float,
) -> list[tuple[Hashable, Hashable, Hashable, Hashable]]:
    """
    Return all unique 4-item tuples whose pair scores are all at least threshold.

    `pair_score` may use (item_a, item_b), (item_b, item_a), or frozenset({item_a, item_b})
    keys. Missing pairs are treated as invalid. Tuple order follows the input order of items.
    Duplicate items are rejected because they make pair identity ambiguous.
    """
    items = list(items)
    item_count = len(items)
    if item_count < 4:
        return []

    neighbors = {item: set() for item in items}
    if len(neighbors) != item_count:
        raise ValueError("`items` must not contain duplicates.")

    def get_score(item_a: Hashable, item_b: Hashable) -> float:
        """Return the score for an unordered pair, or -inf when the pair is missing."""
        if item_a == item_b:
            return float("-inf")
        if (item_a, item_b) in pair_score:
            return pair_score[(item_a, item_b)]
        if (item_b, item_a) in pair_score:
            return pair_score[(item_b, item_a)]
        unordered_pair = frozenset((item_a, item_b))
        if unordered_pair in pair_score:
            return pair_score[unordered_pair]
        return float("-inf")

    forward_neighbors = {item: [] for item in items}
    for i in range(item_count):
        item_a = items[i]
        for j in range(i + 1, item_count):
            item_b = items[j]
            if get_score(item_a, item_b) >= threshold:
                neighbors[item_a].add(item_b)
                neighbors[item_b].add(item_a)
                forward_neighbors[item_a].append(item_b)

    valid_tuples = []
    for item_a in items:
        for item_b in forward_neighbors[item_a]:
            common_ab_neighbors = neighbors[item_a] & neighbors[item_b]
            for item_c in (x for x in forward_neighbors[item_b] if x in common_ab_neighbors):
                common_abc_neighbors = common_ab_neighbors & neighbors[item_c]
                for item_d in (x for x in forward_neighbors[item_c] if x in common_abc_neighbors):
                    valid_tuples.append((item_a, item_b, item_c, item_d))
    return valid_tuples
