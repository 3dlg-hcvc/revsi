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
from utils import calculate_nearest_distance, are_3d_obbs_overlapping, sample_obb_surface_points, write_csv


QUESTION_TYPE = "object_abs_distance"
OBJ_ABS_DISTANCE_QUESTION_TEMPLATE = """
Measuring from the closest point of each object, what is the direct distance between the {object_name_1} and the {object_name_2} (in meters)?
""".strip()

EXCLUDE_OBJ_NAMES = [
    "L-shape sofa", "L-shape couch", "u-shape couch", "L-shape bench", "shower seat", "bed"
]
SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
DISTANCE_SUBSAMPLING_RULES = [
    (1.0, 2.0, 0.8),
    (2.0, 3.0, 0.5),
]
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "used_obj_ids", "gt"]


def _is_valid_object(obj_name, obj_count):
    return obj_count == 1 and obj_name not in EXCLUDE_OBJ_NAMES


def _should_generate_question(distance, distance_threshold, rng):
    if distance < distance_threshold:
        return False
    for min_rule_distance, max_rule_distance, skip_probability in DISTANCE_SUBSAMPLING_RULES:
        if min_rule_distance < distance < max_rule_distance:
            return rng.random() >= skip_probability
    return True


def _process_one_scene(indexed_scene_metadata, distance_threshold, max_samples_per_scene, seed):
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
        if not _is_valid_object(obj_name, obj_count_by_name[obj_name]):
            continue
        obj_obb = o3d.geometry.OrientedBoundingBox(
            center=scene_obj["obb"]["center"], R=scene_obj["obb"]["rotation"], extent=scene_obj["obb"]["extent"]
        )
        obj_obb_mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox=obj_obb)
        surface_points = sample_obb_surface_points(obj_obb_mesh)
        eligible_obj_records.append((obj_name, scene_obj["id"], surface_points, obj_obb))
    if len(eligible_obj_records) < 2:
        return []
    rows = []

    rng.shuffle(eligible_obj_records)
    eligible_obj_pairs = list(itertools.combinations(eligible_obj_records, 2))
    rng.shuffle(eligible_obj_pairs)

    scene_question_count = 0

    for first_obj_record, second_obj_record in eligible_obj_pairs:
        if scene_question_count >= max_samples_per_scene:
            break

        first_obj_name, first_obj_id, first_surface_points, first_obb = first_obj_record
        second_obj_name, second_obj_id, second_surface_points, second_obb = second_obj_record
        nearest_distance = (
            0.0
            if are_3d_obbs_overlapping(first_obb, second_obb)
            else calculate_nearest_distance(first_surface_points, second_surface_points)
        )
        if not _should_generate_question(nearest_distance, distance_threshold, rng):
            continue

        scene_question_count += 1

        rows.append(
            {
                "dataset": scene["dataset"],
                "scene_id": scene["scene_id"],
                "question_type": QUESTION_TYPE,
                "question": OBJ_ABS_DISTANCE_QUESTION_TEMPLATE.format(object_name_1=first_obj_name, object_name_2=second_obj_name),
                "used_obj_ids": [first_obj_id, second_obj_id],
                "gt": f"{nearest_distance:.1f}"
            }
        )
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object absolute distance questions.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    parser.add_argument("--distance_threshold", type=float, default=1, help="Distance threshold.")
    parser.add_argument("--max_samples_per_scene", "--max_sample_per_scene", type=int, default=5, help="Maximum number of samples per scene.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="obj_abs_distance.csv", help="Path to output preprocessed data.")
    args = parser.parse_args()

    with urllib.request.urlopen(SCENE_METADATA_URL) as response:
        scene_metadata = json.load(response)

    scene_rows = process_map(
        partial(
            _process_one_scene,
            distance_threshold=args.distance_threshold,
            max_samples_per_scene=args.max_samples_per_scene,
            seed=args.seed,
        ),
        list(enumerate(scene_metadata)), max_workers=args.num_workers, chunksize=1
    )
    rows = list(itertools.chain(*scene_rows))

    write_csv(args.output_path, rows, CSV_FIELDNAMES)

    print(f"Done. Data saved to {os.path.abspath(args.output_path)}.")
