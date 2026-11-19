
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


def extract_audio_features(
    wav_path: Path,
    checkpoint_path: Path,
    extractor_dir: Path,
) -> np.ndarray:

    wav_path = Path(wav_path)
    checkpoint_path = Path(checkpoint_path)
    extractor_dir = Path(extractor_dir)

    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if str(extractor_dir) not in sys.path:
        sys.path.insert(0, str(extractor_dir))

    import vggish_input
    import vggish_slim

    # Original VGGish audio preprocessing
    audio_input = vggish_input.wavfile_to_examples(
        str(wav_path)
    ).astype(np.float32)

    if audio_input.shape != (10, 96, 64):
        raise ValueError(
            f"Expected (10, 96, 64), got {audio_input.shape}"
        )

    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()

    # Build VGGish and load the original weights
    with tf.Graph().as_default():
        vggish_slim.define_vggish_slim(training=False)

        graph = tf.compat.v1.get_default_graph()

        input_tensor = graph.get_tensor_by_name(
            "vggish/input_features:0"
        )

        output_tensor = graph.get_tensor_by_name(
            "vggish/embedding:0"
        )

        with tf.compat.v1.Session() as sess:
            vggish_slim.load_vggish_slim_checkpoint(
                sess,
                str(checkpoint_path),
            )

            features = sess.run(
                output_tensor,
                feed_dict={input_tensor: audio_input},
            )

    result = features.astype(np.float32)

    if result.shape != (10, 128):
        raise ValueError(
            f"Expected (10, 128), got {result.shape}"
        )

    if not np.isfinite(result).all():
        raise ValueError("Invalid audio features")

    return result
