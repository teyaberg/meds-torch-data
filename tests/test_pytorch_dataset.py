from pathlib import Path

from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.pytorch_dataset import MEDSPytorchDataset


def test_dataset(tensorized_MEDS_dataset: Path):
    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset,
        max_seq_len=10,
    )

    pyd = MEDSPytorchDataset(config, split="train")

    assert len(pyd) == 4, "The dataset should have 4 samples corresponding to the train subjects."
    assert set(pyd.subject_ids) == {239684, 1195293, 68729, 814703}
    assert pyd.max_seq_len == 10

    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert batch is not None


def test_dataset_with_task(tensorized_MEDS_dataset_with_task: tuple[Path, Path, str]):
    cohort_dir, tasks_dir, task_name = tensorized_MEDS_dataset_with_task

    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=cohort_dir,
        task_labels_dir=(tasks_dir / task_name),
        max_seq_len=10,
    )

    pyd = MEDSPytorchDataset(config, split="train")

    assert len(pyd) == 13, "The dataset should have 10 task samples corresponding to the train samples."
    assert pyd.index == [
        (239684, 0, 3),
        (239684, 0, 4),
        (239684, 0, 5),
        (1195293, 0, 3),
        (1195293, 0, 4),
        (1195293, 0, 6),
        (68729, 0, 2),
        (68729, 0, 2),
        (68729, 0, 2),
        (68729, 0, 2),
        (814703, 0, 2),
        (814703, 0, 2),
        (814703, 0, 2),
    ]

    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        assert "boolean_value" in samp, "Each sample in the labeled setting should have the label"

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert batch is not None
    assert "boolean_value" in batch, "The batch should have the label in the labeled setting."
