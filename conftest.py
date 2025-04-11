"""Test set-up and fixtures code."""

import importlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import meds_torchdata.pytest_plugin
from meds_torchdata import MEDSPytorchDataset

importlib.reload(meds_torchdata.pytest_plugin)


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    sample_pytorch_dataset: MEDSPytorchDataset,
    sample_pytorch_dataset_with_task: MEDSPytorchDataset,
    tensorized_MEDS_dataset: Path,
    tensorized_MEDS_dataset_with_task: Path,
    simple_static_MEDS: Path,
    simple_static_MEDS_dataset_with_task: Path,
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "simple_static_MEDS": simple_static_MEDS,
            "simple_static_MEDS_dataset_with_task": simple_static_MEDS_dataset_with_task,
            "tensorized_MEDS_dataset": tensorized_MEDS_dataset,
            "tensorized_MEDS_dataset_with_task": tensorized_MEDS_dataset_with_task,
            "sample_pytorch_dataset": sample_pytorch_dataset,
            "sample_pytorch_dataset_with_task": sample_pytorch_dataset_with_task,
        }
    )
