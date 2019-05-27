from pathlib import Path
import os

PATH_DATAPAPERS_FIXED = os.path.expanduser(
    "~/Documents/Stage/sentences_embeddings/Exports/datapapers_fixed"
)


def test_reproductible_datasets():
    gen_files = sorted(
        Path(PATH_DATAPAPERS_FIXED).glob(f"**/context_2000_300_5_*")
    )

    assert len(gen_files) == 10
