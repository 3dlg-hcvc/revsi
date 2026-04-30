import os
import json
import argparse
import urllib.request
from tqdm import tqdm
from utils import write_csv
from shapely.geometry import Polygon


SCENE_METADATA_URL = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/raw/main/metadata/3d_annotation.json"
QUESTION_TYPE_BY_ROOM_TYPE = {
    "all_room": "room_size_estimation_multiple",
    "single_room": "room_size_estimation_single",
}
CSV_FIELDNAMES = ["dataset", "scene_id", "question_type", "question", "gt"]

MULTIPLE_ROOM_SIZE_TEMPLATE = """
What is the size of the scene (in square meters)? If multiple rooms are shown, estimate the size of the combined space.
""".strip()

SINGLE_ROOM_SIZE_TEMPLATE = """
What is the size of the main room (in square meters)? If multiple rooms are shown, estimate only the size of the dominant room in which the video is primarily recorded.
""".strip()

ROOM_SIZE_TEMPLATE_BY_ROOM_TYPE = {
    "all_room": MULTIPLE_ROOM_SIZE_TEMPLATE,
    "single_room": SINGLE_ROOM_SIZE_TEMPLATE,
}


def _has_boundary_annotation(scene):
    return "scene_area_2d_polygon" in scene and "scene_area_type" in scene


def _get_room_type(scene):
    # all_room is also "multiple_room"
    return "all_room" if scene["scene_area_type"] == "all_room" else "single_room"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate room size estimation questions.")
    parser.add_argument("--output_path", type=str, default="room_size_estimation.csv", help="Path to output preprocessed data.")
    args = parser.parse_args()

    with urllib.request.urlopen(SCENE_METADATA_URL) as response:
        scene_metadata = json.load(response)

    rows = []
    for scene in tqdm(scene_metadata):
        
        if not _has_boundary_annotation(scene):
            continue
        
        room_type = _get_room_type(scene)
        boundary_points_2d = [(point[0], point[1]) for point in scene["scene_area_2d_polygon"]]
        room_polygon_area = Polygon(boundary_points_2d).area
        rows.append(
            {
                "dataset": scene["dataset"],
                "scene_id": scene["scene_id"],
                "question_type": QUESTION_TYPE_BY_ROOM_TYPE[room_type],
                "question": ROOM_SIZE_TEMPLATE_BY_ROOM_TYPE[room_type],
                "gt": f"{room_polygon_area:.1f}"
            }
        )

    write_csv(args.output_path, rows, CSV_FIELDNAMES)

    print(f"Done. Data saved to {os.path.abspath(args.output_path)}.")
