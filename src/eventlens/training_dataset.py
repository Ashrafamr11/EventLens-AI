
from pathlib import Path

from eventlens.llp_dataset import load_clip_labels
from eventlens.llp_features import load_video_features
from eventlens.llp_labels import encode_multilabel


class LLPDataset:
    def __init__(
        self,
        split,
        class_names,
        data_dir="data/raw/llp",
        features_dir="data/processed/feats",
    ):
        self.samples = load_clip_labels(split, data_dir)
        self.class_names = list(class_names)
        self.features_dir = Path(features_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        features = load_video_features(
            sample["video_id"],
            self.features_dir,
        )

        labels = encode_multilabel(
            sample["labels"],
            self.class_names,
        )

        return {
            "audio": features["audio"],
            "visual": features["visual"],
            "motion": features["motion"],
            "labels": labels,
        }
