
import numpy as np

from eventlens.llp_features import load_video_features


def test_load_video_features(tmp_path):
    video_id = "test_video"

    for folder in ["vggish", "res152", "r2plus1d_18"]:
        (tmp_path / folder).mkdir()

    audio = np.ones((10, 128), dtype=np.float32)

    visual = np.repeat(
        np.arange(80, dtype=np.float32)[:, None],
        2048,
        axis=1,
    )

    motion = np.ones((10, 512), dtype=np.float32)

    np.save(tmp_path / "vggish" / f"{video_id}.npy", audio)
    np.save(tmp_path / "res152" / f"{video_id}.npy", visual)
    np.save(tmp_path / "r2plus1d_18" / f"{video_id}.npy", motion)

    features = load_video_features(video_id, tmp_path)

    assert features["audio"].shape == (10, 128)
    assert features["visual"].shape == (10, 2048)
    assert features["motion"].shape == (10, 512)

    expected = np.arange(10, dtype=np.float32) * 8 + 3.5

    np.testing.assert_allclose(
        features["visual"][:, 0],
        expected,
    )
