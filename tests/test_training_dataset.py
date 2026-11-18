
import numpy as np

from eventlens.training_dataset import LLPDataset


def test_training_dataset(tmp_path):
    data_dir = tmp_path / "annotations"
    features_dir = tmp_path / "features"
    data_dir.mkdir()

    (data_dir / "AVVP_train.csv").write_text(
        "filename\tevent_labels\n"
        "BjCEufrlXm4_20_30\tSpeech,Dog\n",
        encoding="utf-8",
    )

    shapes = {
        "vggish": (10, 128),
        "res152": (80, 2048),
        "r2plus1d_18": (10, 512),
    }

    for name, shape in shapes.items():
        folder = features_dir / name
        folder.mkdir(parents=True)

        np.save(
            folder / "BjCEufrlXm4.npy",
            np.ones(shape, dtype=np.float32),
        )

    dataset = LLPDataset(
        split="train",
        class_names=["Cat", "Dog", "Speech"],
        data_dir=data_dir,
        features_dir=features_dir,
    )

    assert len(dataset) == 1

    sample = dataset[0]

    assert sample["audio"].shape == (10, 128)
    assert sample["visual"].shape == (10, 2048)
    assert sample["motion"].shape == (10, 512)

    np.testing.assert_array_equal(
        sample["labels"],
        np.array([0, 1, 1], dtype=np.float32),
    )
