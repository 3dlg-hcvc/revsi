import os
import json
import random
import argparse
import urllib.request
from tqdm import tqdm
from utils import write_csv
from collections import defaultdict


QUESTION_TYPE = "object_size_estimation"
OBJ_SIZE_QUESTION_TEMPLATE = """
Based on visual evidence from the video, what is the length of the longest dimension (length, width, or height) of the {object_name}, measured in centimeters?
""".strip()

EXCLUDE_OBJ_NAMES = [
    "window",
    "bi-fold door",
    "room door",
    "wooden door",
    "wooden room door",
    "room entrance door",
    "light switch",
    "blinds",
    "roller blind",
    "shoes",
    "slippers",
    "clothes",
    "towel",
    "shower towel",
    "jacket",
    "shower curtain",
    "pot",
    "mirror",
    "scissor",
    "sink",
    "mirror",
    "comforter",
    "curtain",
    "backpack",
    "dog",
    "bed",
    "bed frame",
    "adult bed",
    "mattress",
    "bunk beds",
    "toilet",
    "bidet",
    "laptop",
    "smoke detector",
    "tv remote",
    "tv remote control",
    "remote controller",
    "projector remote",
    "football",
    "toilet paper",
    "toilet paper roll",
    "paper towel",
    "paper towel roll",
    "computer mouse",
    "door",
    "intercom",
    "toothpaste",
    "electric kettle",
    "iron",
    "steam iron",
    "range hood",
    "toilet brush",
    "washer",
    "washing machine",
    "dryer",
    "monitor",
    "fireplace",
    "cup",
    "mug",
    "coffee cup",
    "starbucks cup",
    "dryer sheets box",
    "ceiling light",
    "recessed downlight",
    "wall light",
    "shower seat",
    "power strip",
    "keyboard",
    "computer tower",
    "microwave",
    "dishwasher",
    "apron",
    "kitchen apron",
    "potted plant",
    "wall clock",
    "hair dryer",
    "tissue box",
    "telephone",
    "yoga mat",
    "yoga mat roll",
    "table phone",
    "desk phone",
    "range oven",
    "water filter pitcher",
    "headphones",
    "exhaust fan",
    "smartphone",
    "projector",
    "ceiling-mounted projector",
    "thermostat",
]

SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "used_obj_ids", "gt"]
OBJ_SIZE_SUBSAMPLING_RULE_BY_NAME = {
    "radiator": (70, 100, 0.01),
    "chair": (75, 95, 0.05),
    "armchair": (75, 95, 0.05),
    "office chair": (75, 95, 0.05),
    "sofa chair": (75, 95, 0.05),
    "bathtub": (150, 170, 0.005),
    "trash can": (20, 40, 0.05),
    "trash bin": (20, 40, 0.05),
    "refrigerator": (170, 200, 0.05),
    "double-bowl kitchen sink": (70, 90, 0.1),
    "clock": (20, 40, 0.4),
    "wardrobe": (180, 200, 0.1),
    "tv": (-float("inf"), float("inf"), 0.2),
    "oven": (-float("inf"), float("inf"), 0.4),
    "table": (-float("inf"), float("inf"), 0.4),
}


def _is_valid_object(obj_name, obj_count):
    return obj_count == 1 and obj_name not in EXCLUDE_OBJ_NAMES


def _should_generate_question(obj_name, obj_max_extent_cm):
    subsampling_rule = OBJ_SIZE_SUBSAMPLING_RULE_BY_NAME.get(obj_name)
    if subsampling_rule is None:
        return True
    min_extent, max_extent, keep_probability = subsampling_rule
    if min_extent < obj_max_extent_cm < max_extent:
        return random.random() <= keep_probability
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object size estimation questions.")
    parser.add_argument("--max_samples_per_scene", type=int, default=6)
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="obj_size_estimation.csv", help="Path to output preprocessed data.")
    args = parser.parse_args()

    random.seed(args.seed)

    with urllib.request.urlopen(SCENE_METADATA_URL) as response:
        scene_metadata = json.load(response)

    rows = []

    for scene in tqdm(scene_metadata):
        scene_question_count = 0
        obj_count_by_name = defaultdict(int)
        for scene_obj in scene["objects"]:
            obj_count_by_name[scene_obj["name"]] += 1
        shuffled_scene_objects = list(scene["objects"])
        random.shuffle(shuffled_scene_objects)

        for scene_obj in shuffled_scene_objects:
            if scene_question_count >= args.max_samples_per_scene:
                break
            obj_name = scene_obj["name"]
            if not _is_valid_object(obj_name, obj_count_by_name[obj_name]):
                continue

            obj_max_extent_cm = max(scene_obj["obb"]["extent"]) * 100. # convert from meters to centimeters
            if not _should_generate_question(obj_name, obj_max_extent_cm):
                continue

            scene_question_count += 1
            rows.append(
                {
                    "dataset": scene["dataset"],
                    "scene_id": scene["scene_id"],
                    "question_type": QUESTION_TYPE,
                    "question": OBJ_SIZE_QUESTION_TEMPLATE.format(object_name=obj_name),
                    "used_obj_ids": [scene_obj["id"]],
                    "gt": f'{obj_max_extent_cm:.0f}'
                }
            )

    write_csv(args.output_path, rows, CSV_FIELDNAMES)

    print(f"Done. Data saved to {os.path.abspath(args.output_path)}.")
