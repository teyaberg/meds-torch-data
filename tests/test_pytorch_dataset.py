from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch


def test_dataset(sample_pytorch_dataset: MEDSPytorchDataset):
    pyd = sample_pytorch_dataset

    assert len(pyd) == 4, "The dataset should have 4 samples corresponding to the train subjects."

    samps = []
    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        samps.append(samp)

    full_batch = pyd.collate(samps)
    assert isinstance(full_batch, MEDSTorchBatch)

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert isinstance(batch, MEDSTorchBatch)


def test_dataset_with_task(sample_pytorch_dataset_with_task: MEDSPytorchDataset):
    pyd = sample_pytorch_dataset_with_task

    assert len(pyd) == 13, "The dataset should have 10 task samples corresponding to the train samples."
    assert pyd.index == [
        (239684, 3),
        (239684, 4),
        (239684, 5),
        (1195293, 3),
        (1195293, 4),
        (1195293, 6),
        (68729, 2),
        (68729, 2),
        (68729, 2),
        (68729, 2),
        (814703, 2),
        (814703, 2),
        (814703, 2),
    ]

    samps = []
    for i in range(len(pyd)):
        samp = pyd[i]
        assert isinstance(samp, dict), f"Each sample should be a dictionary. For {i} got {type(samp)}"
        assert "boolean_value" in samp, "Each sample in the labeled setting should have the label"
        samps.append(samp)

    full_batch = pyd.collate(samps)
    assert isinstance(full_batch, MEDSTorchBatch)

    dataloader = pyd.get_dataloader(batch_size=32, num_workers=2)
    batch = next(iter(dataloader))
    assert isinstance(batch, MEDSTorchBatch)
