"""Test set-up and fixtures code."""


import subprocess
import tempfile
from pathlib import Path

import pytest
from meds import code_metadata_filepath, subject_splits_filepath

from .preprocessing import MEDS_CODE_METADATA, MEDS_SHARDS, PREPROCESS_SCRIPT, SPLITS_DF


@pytest.fixture(scope="session")
def MEDS_dataset() -> Path:
    with tempfile.TemporaryDirectory() as data_dir:
        data_dir = Path(data_dir)

        for shard, df in MEDS_SHARDS.items():
            fp = data_dir / f"data/{shard}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(fp)

        (data_dir / "metadata").mkdir(parents=True, exist_ok=True)

        MEDS_CODE_METADATA.write_parquet(data_dir / code_metadata_filepath)
        SPLITS_DF.write_parquet(data_dir / subject_splits_filepath)

        yield data_dir


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset(MEDS_dataset: Path) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory() as cohort_dir:
        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={str(MEDS_dataset)}",
            f"output_dir={cohort_dir}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        yield MEDS_dataset, Path(cohort_dir)
