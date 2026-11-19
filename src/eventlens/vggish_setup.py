
from pathlib import Path
from zipfile import ZipFile
import shutil


def prepare_vggish_extractor(
    archive_path: Path,
    output_dir: Path,
) -> Path:

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        "vggish_params.py",
        "vggish_input.py",
        "vggish_slim.py",
        "mel_features.py",
    ]

    with ZipFile(archive_path) as archive:

        for filename in source_files:
            source = archive.read(
                f"feature_extractor/{filename}"
            ).decode("utf-8")

            if filename == "vggish_slim.py":

                replacements = {
                    "import tensorflow as tf":
                        "import tensorflow.compat.v1 as tf",

                    "slim = tf.contrib.slim":
                        "import tf_slim as slim",

                    "return tf.identity(net1, name='embedding')":
                        "return tf.identity(net, name='embedding')",
                }

                for old, new in replacements.items():
                    if old not in source:
                        raise ValueError(
                            f"Expected VGGish code not found: {old}"
                        )
                    source = source.replace(old, new)

            (output_dir / filename).write_text(
                source,
                encoding="utf-8",
            )

        checkpoint_name = "vggish_model.ckpt"
        checkpoint_path = output_dir / checkpoint_name

        info = archive.getinfo(
            f"feature_extractor/{checkpoint_name}"
        )

        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size != info.file_size
        ):
            with archive.open(info) as src:
                with checkpoint_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    return output_dir
