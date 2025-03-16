"""Test set-up and fixtures code."""


import subprocess
import tempfile
from pathlib import Path

import pytest
from meds_testing_helpers.dataset import MEDSDataset

from .preprocessing import PREPROCESS_SCRIPT


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset(simple_static_MEDS: Path) -> Path:
    with tempfile.TemporaryDirectory() as cohort_dir:
        cohort_dir = Path(cohort_dir)

        command = [
            str(PREPROCESS_SCRIPT),
            f"MEDS_dataset_dir={str(simple_static_MEDS)}",
            f"output_dir={str(cohort_dir)}",
        ]

        out = subprocess.run(" ".join(command), shell=True, check=False, capture_output=True)

        error_str = (
            f"Command failed with return code {out.returncode}.\n"
            f"Command stdout:\n{out.stdout.decode()}\n"
            f"Command stderr:\n{out.stderr.decode()}"
        )

        assert out.returncode == 0, error_str

        yield cohort_dir


@pytest.fixture(scope="session")
def tensorized_MEDS_dataset_with_task(
    tensorized_MEDS_dataset: Path,
    simple_static_MEDS_dataset_with_task: Path,
) -> tuple[Path, Path, str]:
    cohort_dir = tensorized_MEDS_dataset

    D = MEDSDataset(root_dir=simple_static_MEDS_dataset_with_task)

    if len(D.task_names) != 1:
        raise ValueError("Expected only one task in the dataset.")

    yield cohort_dir, D.task_root_dir, D.task_names[0]
