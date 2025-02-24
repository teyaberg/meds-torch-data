from pathlib import Path

from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.pytorch_dataset import MEDSPytorchDataset


def test_dataset(tensorized_MEDS_dataset: tuple[Path, Path]):
    config = MEDSTorchDataConfig(
        tensorized_cohort_dir=tensorized_MEDS_dataset[1],
        max_seq_len=100,
    )

    pyd = MEDSPytorchDataset(config, split="train")

    for i in range(len(pyd)):
        pyd[i]

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert batch is not None
