import os
import json
import math
import random
import argparse
import itertools
import numpy as np
import open3d as o3d
import urllib.request
from functools import partial
from collections import defaultdict
from scipy.spatial import cKDTree
from tqdm.contrib.concurrent import process_map
from utils import nearest_distance_to_tree, are_3d_obbs_overlapping, are_xy_projections_overlapping, sample_obb_surface_points, write_csv


QUESTION_TYPE_PREFIX = "object_rel_direction"
SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
QUESTION_LEVELS = ("forward_easy", "backward_easy", "forward_hard", "backward_hard")
EASY_QUESTION_LEVELS = ("forward_easy", "backward_easy")
HARD_QUESTION_LEVELS = ("forward_hard", "backward_hard")
BACKWARD_QUESTION_LEVELS = ("backward_easy", "backward_hard")
QUESTION_TEMPLATE_BY_LEVEL = {
    "forward_easy": "If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my left, right, or back?\nAn object is to my back if I would have to turn at least 135 degrees in order to face it.",
    "backward_easy": "If I am standing by the {positioning_object} and facing in the opposite direction of the {orienting_object}, is the {querying_object} to my left, right, or back?\nAn object is to my back if I would have to turn at least 135 degrees in order to face it.",
    "forward_hard": "If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my front-left, front-right, back-left, or back-right?",
    "backward_hard": "If I am standing by the {positioning_object} and facing in the opposite direction of the {orienting_object}, is the {querying_object} to my front-left, front-right, back-left, or back-right?",

}

OPTIONS_BY_LEVEL = {
    "forward_easy": ["left", "right", "back"],
    "backward_easy": ["left", "right", "back"],
    "forward_hard": ["front-left", "front-right", "back-left", "back-right"],
    "backward_hard": ["front-left", "front-right", "back-left", "back-right"],
}

OPTION_LABEL_BY_INDEX = {0: "A", 1: "B", 2: "C", 3: "D"}
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "used_obj_ids", "options", "gt"]

MAX_OBJECT_XY_EXTENT = 1.4
MIN_OBJECT_SEPARATION = 1.5
MIN_ANGLE_FROM_BOUNDARY = 35


def _is_valid_object(obj, obj_count):
    return (
        obj_count == 1
        and obj["obb"]["extent"][0] <= MAX_OBJECT_XY_EXTENT
        and obj["obb"]["extent"][1] <= MAX_OBJECT_XY_EXTENT
    )


def _get_question_type(question_level):
    return f"{QUESTION_TYPE_PREFIX}_{question_level}"


def _calculate_angle(reference_vectors, target_vectors):
    # dot products
    dot_products = (reference_vectors * target_vectors).sum(axis=1)
    reference_magnitudes = np.linalg.norm(reference_vectors, axis=1)
    target_magnitudes = np.linalg.norm(target_vectors, axis=1)
    denom = np.maximum(reference_magnitudes * target_magnitudes, np.finfo(reference_magnitudes.dtype).eps)
    cos_theta = dot_products / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    cross_products = reference_vectors[0, 0] * target_vectors[:, 1] - reference_vectors[0, 1] * target_vectors[:, 0]
    angles = np.where(cross_products >= 0.0, angles, 2 * math.pi - angles)
    return np.degrees(angles)


def _get_gt(positioning_obj, orienting_obj, querying_obj, level="forward_easy"):
    positioning_center = positioning_obj.center[None, ]
    orienting_center = orienting_obj.center[None, ]
    querying_center = querying_obj.center[None, ]

    querying_vertices = np.asarray(querying_obj.get_box_points())
    querying_points = np.concatenate([querying_center, querying_vertices], axis=0)

    orienting_vec = orienting_center - positioning_center
    if level in BACKWARD_QUESTION_LEVELS:
        orienting_vec = -orienting_vec

    querying_vecs = querying_points - positioning_center
    querying_angles = _calculate_angle(orienting_vec[:, :2], querying_vecs[:, :2])

    if level in EASY_QUESTION_LEVELS:
        quadrants = np.digitize(querying_angles, bins=[0, 135, 225, 360])
        quadrant_of_centroid = quadrants[0]
        quadrant_of_vertices = quadrants[1:]

        if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2:
            return None
        boundaries = np.array([0, 135, 225, 360])
        if np.abs(querying_angles[0] - boundaries).min() < MIN_ANGLE_FROM_BOUNDARY:
            return None
    elif level in HARD_QUESTION_LEVELS:
        quadrant_of_centroid = (querying_angles // 90)[0]
        quadrant_of_vertices = (querying_angles // 90)[1:]
        if (quadrant_of_centroid != quadrant_of_vertices).sum() > 2:
            return None
        if min(querying_angles[0] % 90, 90 - (querying_angles[0] % 90)) < MIN_ANGLE_FROM_BOUNDARY:
            return None

    if level in EASY_QUESTION_LEVELS:
        if quadrants[0] == 1:
            gt = "left"
        elif quadrants[0] == 2:
            gt = "back"
        elif quadrants[0] == 3:
            gt = "right"
        else:
            raise ValueError
    elif level in HARD_QUESTION_LEVELS:
        if querying_angles[0] >= 270:
            gt = "front-right"
        elif querying_angles[0] >= 180:
            gt = "back-right"
        elif querying_angles[0] >= 90:
            gt = "back-left"
        else:
            gt = "front-left"
    return gt


def _process_one_scene(indexed_scene_metadata, max_samples_per_scene, seed):
    scene_index, scene = indexed_scene_metadata
    # Use a per-scene seed so multiprocessing worker scheduling does not affect results.
    scene_seed = (seed + scene_index) % (2 ** 32)
    rng = random.Random(scene_seed)
    np.random.seed(scene_seed)

    obj_count_by_name = defaultdict(int)
    for scene_obj in scene["objects"]:
        obj_count_by_name[scene_obj["name"]] += 1

    eligible_obj_records = []
    for scene_obj in scene["objects"]:
        obj_name = scene_obj["name"]
        if not _is_valid_object(scene_obj, obj_count_by_name[obj_name]):
            continue
        obj_obb = o3d.geometry.OrientedBoundingBox(
            center=scene_obj["obb"]["center"], R=scene_obj["obb"]["rotation"], extent=scene_obj["obb"]["extent"]
        )
        obj_obb_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox=obj_obb)
        surface_points = sample_obb_surface_points(obj_obb_mesh)

        surface_tree_3d = cKDTree(surface_points)
        surface_tree_2d = cKDTree(surface_points[:, :2])

        eligible_obj_records.append(
            (
                obj_name,
                scene_obj["id"],
                obj_obb,
                surface_points,
                surface_tree_3d,
                surface_tree_2d,
            )
        )

    if len(eligible_obj_records) < 3:
        return []

    rows = []
    all_obj_triplets = list(itertools.combinations(eligible_obj_records, 3))
    for question_level in QUESTION_LEVELS:
        shuffled_obj_triplets = all_obj_triplets.copy()
        rng.shuffle(shuffled_obj_triplets)
        scene_question_count = 0
        for obj_triplet in shuffled_obj_triplets:
            if scene_question_count >= max_samples_per_scene:
                break
            obj_triplet = list(obj_triplet)
            rng.shuffle(obj_triplet)
            positioning_obj, orienting_obj, querying_obj = obj_triplet
            obj_pairs = (
                (positioning_obj, orienting_obj),
                (positioning_obj, querying_obj),
                (orienting_obj, querying_obj),
            )

            if any(are_3d_obbs_overlapping(first_obj[2], second_obj[2]) for first_obj, second_obj in obj_pairs):
                continue

            if any(are_xy_projections_overlapping(first_obj[2], second_obj[2]) for first_obj, second_obj in obj_pairs):
                continue

            min_dist_3d = min(
                nearest_distance_to_tree(positioning_obj[3], orienting_obj[4]),
                nearest_distance_to_tree(positioning_obj[3], querying_obj[4]),
                nearest_distance_to_tree(orienting_obj[3], querying_obj[4]),
            )
            if min_dist_3d < MIN_OBJECT_SEPARATION:
                continue

            min_dist_2d = min(
                nearest_distance_to_tree(positioning_obj[3][:, :2], orienting_obj[5]),
                nearest_distance_to_tree(positioning_obj[3][:, :2], querying_obj[5]),
                nearest_distance_to_tree(orienting_obj[3][:, :2], querying_obj[5]),
            )
            if min_dist_2d < MIN_OBJECT_SEPARATION:
                continue

            ground_truth = _get_gt(positioning_obj[2], orienting_obj[2], querying_obj[2], level=question_level)
            if ground_truth is None:
                continue

            options = OPTIONS_BY_LEVEL[question_level].copy()
            rng.shuffle(options)
            rows.append(
                {
                    "dataset": scene["dataset"],
                    "scene_id": scene["scene_id"],
                    "question_type": _get_question_type(question_level),
                    "question": QUESTION_TEMPLATE_BY_LEVEL[question_level].format(
                        positioning_object=positioning_obj[0], orienting_object=orienting_obj[0], querying_object=querying_obj[0]
                    ),
                    "used_obj_ids": [positioning_obj[1], orienting_obj[1], querying_obj[1]],
                    "options": [f"{OPTION_LABEL_BY_INDEX[i]}. {obj_name_in_option}" for i, obj_name_in_option in enumerate(options)],
                    "gt": OPTION_LABEL_BY_INDEX[options.index(ground_truth)]
                }
            )
            scene_question_count += 1
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object relative direction questions.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    parser.add_argument("--max_samples_per_scene", "--max_sample_per_scene", type=int, default=4, help="Maximum number of samples per scene.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="obj_rel_direction.csv", help="Path to output preprocessed data.")
    args = parser.parse_args()

    with urllib.request.urlopen(SCENE_METADATA_URL) as response:
        scene_metadata = json.load(response)

    scene_rows = process_map(
        partial(_process_one_scene, max_samples_per_scene=args.max_samples_per_scene, seed=args.seed),
        list(enumerate(scene_metadata)), max_workers=args.num_workers, chunksize=1
    )
    rows = list(itertools.chain(*scene_rows))

    write_csv(args.output_path, rows, CSV_FIELDNAMES)

    print(f"Done. Data saved to {os.path.abspath(args.output_path)}.")
