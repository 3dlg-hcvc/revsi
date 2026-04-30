import os
import json
import random
import argparse
import urllib.request
from tqdm import tqdm
from utils import write_csv
from collections import defaultdict


QUESTION_TYPE = "object_counting_single"
OBJ_COUNT_QUESTION_TEMPLATE = "How many {object_name}(s) are in the scene?"
EXCLUDE_OBJ_NAMES = ["window", "bi-fold door", "light switch", "blinds", "roller blind", "shoes", "slippers", "clothes"]
CHALLENGING_OBJ_NAMES = ["nightstand", "chair", "trash can", "door", "table", "pillow", "table lamp", "backpack", "desk"]
SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
CHALLENGING_OBJ_KEEP_PROBABILITY_BY_COUNT = {1: 0.4, 2: 0.05}
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "used_obj_ids", "gt"]


def _should_generate_question(obj_name, obj_count):
    if obj_name in EXCLUDE_OBJ_NAMES:
        return False
    if obj_name not in CHALLENGING_OBJ_NAMES:
        return obj_count > 1
    keep_probability = CHALLENGING_OBJ_KEEP_PROBABILITY_BY_COUNT.get(obj_count)  # subsampling
    return keep_probability is None or random.random() <= keep_probability


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object counting questions.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="obj_count_single.csv", help="Path to output preprocessed data.")
    args = parser.parse_args()

    random.seed(args.seed)

    with urllib.request.urlopen(SCENE_METADATA_URL) as response:
        scene_metadata = json.load(response)

    rows = []
    for scene in tqdm(scene_metadata):
        obj_count_by_name = defaultdict(int)
        obj_ids_by_name = defaultdict(list)
        for scene_obj in scene["objects"]:
            obj_count_by_name[scene_obj["name"]] += 1
            obj_ids_by_name[scene_obj["name"]].append(scene_obj["id"])
        for obj_name, obj_count in obj_count_by_name.items():
            if not _should_generate_question(obj_name, obj_count):
                continue
            rows.append(
                {
                    "dataset": scene["dataset"],
                    "scene_id": scene["scene_id"],
                    "question_type": QUESTION_TYPE,
                    "question": OBJ_COUNT_QUESTION_TEMPLATE.format(object_name=obj_name),
                    "used_obj_ids": sorted(obj_ids_by_name[obj_name]),
                    "gt": str(obj_count)
                }
            )

    write_csv(args.output_path, rows, CSV_FIELDNAMES)

    print(f"Done. Data saved to {os.path.abspath(args.output_path)}.")

