import os
import json
import random
import argparse
import itertools
import urllib.request
from tqdm import tqdm
from utils import write_csv
from collections import defaultdict


QUESTION_TYPE = "object_counting_multiple"
OBJ_COUNT_QUESTION_TEMPLATE = "How many {object_name_1}(s) and {object_name_2}(s) are in the scene in total?"
EXCLUDE_OBJ_NAMES = ["window", "bi-fold door", "light switch", "blinds", "roller blind", "shoes", "slippers", "clothes"]
CHALLENGING_OBJ_NAMES = ["nightstand", "chair", "trash can", "door", "table", "pillow", "table lamp", "backpack", "desk"]
SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
CHALLENGING_PAIR_KEEP_PROBABILITY = 0.3
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "used_obj_ids", "gt"]


def _is_valid_object(obj_name, obj_count):
    return obj_count > 1 and obj_name not in EXCLUDE_OBJ_NAMES


def _should_generate_question(first_obj_name, first_obj_count, second_obj_name, second_obj_count):
    if first_obj_name not in CHALLENGING_OBJ_NAMES or second_obj_name not in CHALLENGING_OBJ_NAMES:
        return True
    if first_obj_count + second_obj_count != 4:
        return True
    return random.random() <= CHALLENGING_PAIR_KEEP_PROBABILITY


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object counting questions.")
    parser.add_argument("--max_samples_per_scene", type=int, default=3)
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="obj_count_multiple.csv", help="Path to output qa data.")
    args = parser.parse_args()

    random.seed(args.seed)

    with urllib.request.urlopen(SCENE_METADATA_URL) as response:
        scene_metadata = json.load(response)

    rows = []
    for scene in tqdm(scene_metadata):
        scene_question_count = 0
        obj_count_by_name = defaultdict(int)
        obj_ids_by_name = defaultdict(list)
        for scene_obj in scene["objects"]:
            obj_count_by_name[scene_obj["name"]] += 1
            obj_ids_by_name[scene_obj["name"]].append(scene_obj["id"])

        valid_obj_names = []
        for obj_name, obj_count in obj_count_by_name.items():
            if not _is_valid_object(obj_name, obj_count):
                continue
            valid_obj_names.append(obj_name)
        if len(valid_obj_names) < 2:
            continue

        random.shuffle(valid_obj_names)
        valid_obj_name_pairs = list(itertools.combinations(valid_obj_names, 2))
        random.shuffle(valid_obj_name_pairs)

        for first_obj_name, second_obj_name in valid_obj_name_pairs:
            if scene_question_count >= args.max_samples_per_scene:
                break

            first_obj_count = obj_count_by_name[first_obj_name]
            second_obj_count = obj_count_by_name[second_obj_name]

            if not _should_generate_question(first_obj_name, first_obj_count, second_obj_name, second_obj_count):
                continue

            scene_question_count += 1
            rows.append(
                {
                    "dataset": scene["dataset"],
                    "scene_id": scene["scene_id"],
                    "question_type": QUESTION_TYPE,
                    "question": OBJ_COUNT_QUESTION_TEMPLATE.format(object_name_1=first_obj_name, object_name_2=second_obj_name),
                    "used_obj_ids": sorted(obj_ids_by_name[first_obj_name] + obj_ids_by_name[second_obj_name]),
                    "gt": str(first_obj_count + second_obj_count)
                }
            )

    write_csv(args.output_path, rows, CSV_FIELDNAMES)

    print(f"Done. Data saved to {os.path.abspath(args.output_path)}.")
