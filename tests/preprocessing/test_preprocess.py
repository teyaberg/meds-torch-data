"""Tests the full, multi-stage pre-processing pipeline. Only checks tokenized and tensorized outputs."""


import subprocess
import tempfile
from pathlib import Path

import polars as pl

from . import PREPROCESS_SCRIPT, assert_df_equal, check_NRT_output
from .test_tensorization import WANT_NRTS
from .test_tokenization import WANT_SCHEMAS


def test_preprocess(tensorized_MEDS_dataset: tuple[Path, Path]):
    cohort_dir = tensorized_MEDS_dataset

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


def test_preprocess_error_case():
    with tempfile.TemporaryDirectory() as root_dir:
        non_existent_dir = Path(root_dir) / "non_existent_dir"
        cohort_dir = Path(root_dir) / "cohort_dir"

        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={str(non_existent_dir.resolve())}",
            f"output_dir={str(cohort_dir.resolve())}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        assert out.returncode == 1
