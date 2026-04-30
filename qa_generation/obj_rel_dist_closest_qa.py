import os
import json
import random
import argparse
import itertools
import numpy as np
import open3d as o3d
import urllib.request
from functools import partial
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from utils import (
    calculate_nearest_distance,
    find_valid_4_tuples_from_pair_scores,
    are_3d_obbs_overlapping,
    sample_obb_surface_points,
    write_csv,
)


QUESTION_TYPE = "object_rel_distance_closest"
OBJ_REL_DISTANCE_QUESTION_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the {object_name}?
""".strip()

OPTION_LABEL_BY_INDEX = {0: "A", 1: "B", 2: "C", 3: "D"}
SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "used_obj_ids", "options", "gt"]

MIN_OPTION_DISTANCE_DIFF_TO_PRIMARY = 0.3
MIN_OPTION_DISTANCE_TO_PRIMARY = 1


EXCLUDE_OBJ_NAMES = [
    "ceiling light", "ceiling lamp", "bathroom ceiling heater", "ceiling fan with light", "ceiling fan",
    "recessed downlight", "L-shape sofa", "L-shape couch", "u-shape couch", "L-shape bench", "shower seat", "bed"
]


def _is_valid_object(obj_name):
    return obj_name not in EXCLUDE_OBJ_NAMES


def _process_one_scene(indexed_scene_metadata, max_samples_per_scene, seed):
    scene_index, scene = indexed_scene_metadata
    # Use a per-scene seed so multiprocessing worker scheduling does not affect results.
    scene_seed = (seed + scene_index) % (2 ** 32)
    rng = random.Random(scene_seed)
    np.random.seed(scene_seed)

    rows = []
    obj_count_by_name = defaultdict(int)

    for scene_obj in scene["objects"]:
        obj_count_by_name[scene_obj["name"]] += 1

    single_instance_obj_names = []
    multi_instance_obj_names = []
    obj_id_to_geometry = {}
    obj_name_to_ids = defaultdict(list)
    for scene_obj in scene["objects"]:
        obj_name = scene_obj["name"]
        if not _is_valid_object(obj_name):
            continue
        obj_obb = o3d.geometry.OrientedBoundingBox(
            center=scene_obj["obb"]["center"], R=scene_obj["obb"]["rotation"], extent=scene_obj["obb"]["extent"]
        )
        obj_obb_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox=obj_obb)
        surface_points = sample_obb_surface_points(obj_obb_mesh)
        obj_id_to_geometry[scene_obj["id"]] = (surface_points, obj_obb)
        obj_name_to_ids[obj_name].append(scene_obj["id"])
        if obj_count_by_name[obj_name] > 1 and obj_name not in multi_instance_obj_names:
            multi_instance_obj_names.append(obj_name)
        elif obj_count_by_name[obj_name] == 1:
            single_instance_obj_names.append(obj_name)
    if len(single_instance_obj_names) == 0 or len(single_instance_obj_names) + len(multi_instance_obj_names) < 5:
        return []

    # Compute nearest distance for all pairs of object names (single + multi)
    all_obj_names = single_instance_obj_names + multi_instance_obj_names
    obj_pair_distance_by_name = {}
    for (name_a, name_b) in itertools.combinations(all_obj_names, 2):
        min_pair_distance = float("inf")
        for first_obj_id in obj_name_to_ids[name_a]:
            for second_obj_id in obj_name_to_ids[name_b]:
                first_surface_points, first_obb = obj_id_to_geometry[first_obj_id]
                second_surface_points, second_obb = obj_id_to_geometry[second_obj_id]
                if are_3d_obbs_overlapping(first_obb, second_obb):
                    nearest_distance = 0.0
                else:
                    nearest_distance = calculate_nearest_distance(first_surface_points, second_surface_points)
                min_pair_distance = min(min_pair_distance, nearest_distance)
        obj_pair_distance_by_name[(name_a, name_b)] = obj_pair_distance_by_name[(name_b, name_a)] = min_pair_distance

    rng.shuffle(single_instance_obj_names)
    for primary_obj_name in single_instance_obj_names:
        if len(rows) >= max_samples_per_scene:
            break

        option_obj_candidates = [obj_name for obj_name in single_instance_obj_names if obj_name != primary_obj_name] + multi_instance_obj_names
        valid_option_obj_names = [
            option_obj_name
            for option_obj_name in option_obj_candidates
            if obj_pair_distance_by_name[(primary_obj_name, option_obj_name)] > MIN_OPTION_DISTANCE_TO_PRIMARY
        ]

        option_pair_primary_distance_diffs = {}
        for pair in itertools.combinations(valid_option_obj_names, 2):
            dist_diff = abs(
                obj_pair_distance_by_name[(primary_obj_name, pair[0])] - obj_pair_distance_by_name[(primary_obj_name, pair[1])]
            )
            if dist_diff > MIN_OPTION_DISTANCE_DIFF_TO_PRIMARY:
                option_pair_primary_distance_diffs[pair] = dist_diff

        valid_4_tuples = find_valid_4_tuples_from_pair_scores(
            items=valid_option_obj_names,
            pair_score=option_pair_primary_distance_diffs,
            threshold=MIN_OPTION_DISTANCE_DIFF_TO_PRIMARY,
        )
        if not valid_4_tuples:
            continue
        option_obj_names = rng.choice(valid_4_tuples)
        option_obj_names = list(option_obj_names)
        rng.shuffle(option_obj_names)
        option_dists_to_primary = [obj_pair_distance_by_name[(primary_obj_name, option_obj_name)] for option_obj_name in option_obj_names]
        option_dists_to_primary = np.array(option_dists_to_primary, dtype=np.float32)

        used_obj_ids = [obj_name_to_ids[primary_obj_name][0]]
        for option_obj_name in option_obj_names:
            used_obj_ids += obj_name_to_ids[option_obj_name]

        rows.append(
            {
                "dataset": scene["dataset"],
                "scene_id": scene["scene_id"],
                "question_type": QUESTION_TYPE,
                "question": OBJ_REL_DISTANCE_QUESTION_TEMPLATE.format(
                    choice_a=option_obj_names[0], choice_b=option_obj_names[1], choice_c=option_obj_names[2], choice_d=option_obj_names[3],
                    object_name=primary_obj_name
                ),
                "used_obj_ids": used_obj_ids,
                "options": [f"{OPTION_LABEL_BY_INDEX[i]}. {obj_name_in_option}" for i, obj_name_in_option in enumerate(option_obj_names)],
                "gt": OPTION_LABEL_BY_INDEX[np.argmin(option_dists_to_primary)]
            }
        )
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object relative distance questions.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    parser.add_argument("--max_samples_per_scene", "--max_sample_per_scene", type=int, default=9, help="Maximum number of samples per scene.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="obj_rel_dist_closest.csv", help="Path to output preprocessed data.")
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
