"""Tests the full, multi-stage pre-processing pipeline. Only checks tokenized and tensorized outputs."""


import subprocess
import tempfile
from pathlib import Path

import polars as pl
from meds import code_metadata_filepath, subject_splits_filepath

from . import (
    MEDS_CODE_METADATA,
    MEDS_SHARDS,
    PREPROCESS_SCRIPT,
    SPLITS_DF,
    assert_df_equal,
    check_NRT_output,
)
from .test_tensorization import WANT_NRTS
from .test_tokenization import WANT_SCHEMAS


def test_preprocess():
    with tempfile.TemporaryDirectory() as root_dir:
        root_dir = Path(root_dir)

        data_dir = root_dir / "MEDS_dataset"
        for shard, df in MEDS_SHARDS.items():
            fp = data_dir / f"data/{shard}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(fp)

        (data_dir / "metadata").mkdir(parents=True, exist_ok=True)

        MEDS_CODE_METADATA.write_parquet(data_dir / code_metadata_filepath)
        SPLITS_DF.write_parquet(data_dir / subject_splits_filepath)

        cohort_dir = root_dir / "cohort"
        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={str(data_dir)}",
            f"output_dir={str(cohort_dir)}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        cohort_dir_contents = list(cohort_dir.rglob("*.parquet")) + list(cohort_dir.rglob("*.nrt"))
        cohort_dir_contents_str = "\n".join(f"  - {f.relative_to(cohort_dir)}" for f in cohort_dir_contents)

        for shard, want_schema in WANT_SCHEMAS.items():
            fp = cohort_dir / f"tokenization/{shard}.parquet"
            err_str = f"Expected output file {fp} not found. Directory contents:\n" + cohort_dir_contents_str

            assert fp.exists(), err_str
            got_schema = pl.read_parquet(fp)
            assert_df_equal(got_schema, want_schema, check_column_order=False)

        for shard, want_NRT in WANT_NRTS.items():
            fp = cohort_dir / f"data/{shard}"
            err_str = f"Expected output file {fp} not found. Directory contents:\n" + cohort_dir_contents_str

            assert fp.exists(), err_str
            check_NRT_output(fp, want_NRT, f"{shard} NRT differs!")
