from swift.dataset import (
    ResponsePreprocessor, DatasetMeta, register_dataset, SubsetDataset, HfDataset, MediaResource
)
from typing import Optional, Dict, Any
import os


NQ_TYPES = [
    "object_counting_single",
    "object_counting_multiple",
    "object_size_estimation",
    "object_abs_distance",
    "room_size_estimation_single",
    "room_size_estimation_multiple",
]


class ReVSIPreprocessor(ResponsePreprocessor):
    video_url = "https://huggingface.co/datasets/3dlg-hcvc/ReVSI/resolve/main/video.zip"

    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        self.nframes = kwargs.pop("nframes", "all")
        super().__init__(columns=columns, **kwargs)

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        local_alias = f"revsi_video"
        self.local_dirs = MediaResource.download(self.video_url, local_alias=local_alias, file_type="compressed")
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pre_prompt = "These are frames of a video."
        if row["question_type"] in NQ_TYPES:
            post_prompt = "Answer the question using a single integer or decimal number."
            query = row["query"]
        else:
            options_str = "Options:\n" + "\n".join(row["options"])
            post_prompt = "Answer with the option's letter from the given choices directly."
            query = f"{row['query']}\n{options_str}"
        query = pre_prompt + "\n" + query + "\n" + post_prompt
        data = {
            "videos": os.path.join(self.local_dirs, f"{self.nframes}_frame", f"{row['scene_id']}.mp4"),
            "query": query,
            "response": row["ground_truth"],
            "id": row["id"],
        }
        return super().preprocess(data)


register_dataset(
    DatasetMeta(
        hf_dataset_id='3dlg-hcvc/ReVSI',
        subsets=[
            SubsetDataset(subset="all_frame", split=["test"], preprocess_func=ReVSIPreprocessor(nframes="all")),
            SubsetDataset(subset="16_frame", split=["test"], preprocess_func=ReVSIPreprocessor(nframes=16)),
            SubsetDataset(subset="32_frame", split=["test"], preprocess_func=ReVSIPreprocessor(nframes=32)),
            SubsetDataset(subset="64_frame", split=["test"], preprocess_func=ReVSIPreprocessor(nframes=64)),
        ],
        tags=['en', 'multi-modal', 'video', '3d', 'vqa'],
        huge_dataset=True
    )
)
