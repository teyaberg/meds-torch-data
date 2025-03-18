# MEDS TorchData: A PyTorch Dataset Class for MEDS Datasets

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Python 3.11+](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![PyPI](https://img.shields.io/badge/PyPI-v0.0.1a1-blue?logoColor=blue)](https://pypi.org/project/meds-torch-data/)
[![Documentation Status](https://readthedocs.org/projects/meds-testing-helpers/badge/?version=latest)](https://meds-testing-helpers.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/mmcdermott/meds-torch-data/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/meds-torch-data/actions/workflows/tests.yaml)
[![Test Coverage](https://codecov.io/github/mmcdermott/meds-torch-data/graph/badge.svg?token=BV119L5JQJ)](https://codecov.io/github/mmcdermott/meds-torch-data)
[![Code Quality](https://github.com/mmcdermott/meds-torch-data/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/meds-torch-data/actions/workflows/code-quality-main.yaml)
[![Hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![Contributors](https://img.shields.io/github/contributors/oufattole/meds-torch.svg)](https://github.com/mmcdermott/meds-torch-data/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/meds-torch-data/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/meds-torch-data#license)

## 🚀 Quick Start

### Step 1: Install

```bash
pip install meds-torch-data
```

### Step 2: Data Tensorization:

> [!WARNING]
> If your dataset is not sharded by split, you need to run a reshard to split stage first! You can enable this by adding the `do_reshard=True` argument to the command below.

```bash
MEDS_tensorize input_dir=... output_dir=...
```

### Step 3: Use the dataset:

In your code, simply:

```python
from meds_torchdata import MEDSPytorchDataset

pyd = MEDSPytorchDataset(...)
```

To see how this works, let's look at some examples. These examples will be powered by some synthetic data
defined as "fixtures" in this package's pytest stack; namely, we'll use the following fixtures:

- `simple_static_MEDS`: This will point to a Path containing a simple MEDS dataset.
- `simple_static_MEDS_dataset_with_task`: This will point to a Path containing a simple MEDS dataset
    with a boolean-value task defined. The core data is the same between both the `simple_static_MEDS` and
    this dataset, but the latter has a task defined.
- `tensorized_MEDS_dataset` fixture that points to a Path containing the tensorized and schema files for
    the `simple_static_MEDS` dataset.
- `tensorized_MEDS_dataset_with_task` fixture that points to a Path containing the tensorized and schema
    files for the `simple_static_MEDS_dataset_with_task` dataset.

You can find these in either the [`conftest.py`](conftest.py) file for this repository or the
[`meds_testing_helpers`](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers) package, which
this package leverages for testing.

To start, let's take a look at this syntehtic data. It is sharded by split, and we'll look at the train split
first, which has two shards (we convert to polars just for prettier printing). It has four subjects
across the two shards.

```python
>>> import polars as pl
>>> from meds_testing_helpers.dataset import MEDSDataset
>>> D = MEDSDataset(root_dir=simple_static_MEDS)
>>> train_0 = pl.from_arrow(D.data_shards["train/0"])
>>> train_0
shape: (30, 4)
┌────────────┬─────────────────────┬────────────────────┬───────────────┐
│ subject_id ┆ time                ┆ code               ┆ numeric_value │
│ ---        ┆ ---                 ┆ ---                ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ str                ┆ f32           │
╞════════════╪═════════════════════╪════════════════════╪═══════════════╡
│ 239684     ┆ null                ┆ EYE_COLOR//BROWN   ┆ null          │
│ 239684     ┆ null                ┆ HEIGHT             ┆ 175.271118    │
│ 239684     ┆ 1980-12-28 00:00:00 ┆ DOB                ┆ null          │
│ 239684     ┆ 2010-05-11 17:41:51 ┆ ADMISSION//CARDIAC ┆ null          │
│ 239684     ┆ 2010-05-11 17:41:51 ┆ HR                 ┆ 102.599998    │
│ …          ┆ …                   ┆ …                  ┆ …             │
│ 1195293    ┆ 2010-06-20 20:24:44 ┆ HR                 ┆ 107.699997    │
│ 1195293    ┆ 2010-06-20 20:24:44 ┆ TEMP               ┆ 100.0         │
│ 1195293    ┆ 2010-06-20 20:41:33 ┆ HR                 ┆ 107.5         │
│ 1195293    ┆ 2010-06-20 20:41:33 ┆ TEMP               ┆ 100.400002    │
│ 1195293    ┆ 2010-06-20 20:50:04 ┆ DISCHARGE          ┆ null          │
└────────────┴─────────────────────┴────────────────────┴───────────────┘
>>> train_1 = pl.from_arrow(D.data_shards["train/1"])
>>> train_1
shape: (14, 4)
┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
│ subject_id ┆ time                ┆ code                  ┆ numeric_value │
│ ---        ┆ ---                 ┆ ---                   ┆ ---           │
│ i64        ┆ datetime[μs]        ┆ str                   ┆ f32           │
╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
│ 68729      ┆ null                ┆ EYE_COLOR//HAZEL      ┆ null          │
│ 68729      ┆ null                ┆ HEIGHT                ┆ 160.395309    │
│ 68729      ┆ 1978-03-09 00:00:00 ┆ DOB                   ┆ null          │
│ 68729      ┆ 2010-05-26 02:30:56 ┆ ADMISSION//PULMONARY  ┆ null          │
│ 68729      ┆ 2010-05-26 02:30:56 ┆ HR                    ┆ 86.0          │
│ …          ┆ …                   ┆ …                     ┆ …             │
│ 814703     ┆ 1976-03-28 00:00:00 ┆ DOB                   ┆ null          │
│ 814703     ┆ 2010-02-05 05:55:39 ┆ ADMISSION//ORTHOPEDIC ┆ null          │
│ 814703     ┆ 2010-02-05 05:55:39 ┆ HR                    ┆ 170.199997    │
│ 814703     ┆ 2010-02-05 05:55:39 ┆ TEMP                  ┆ 100.099998    │
│ 814703     ┆ 2010-02-05 07:02:30 ┆ DISCHARGE             ┆ null          │
└────────────┴─────────────────────┴───────────────────────┴───────────────┘
>>> sorted(set(train_0["subject_id"].unique()) | set(train_1["subject_id"].unique()))
[68729, 239684, 814703, 1195293]

```

Given this data, when we build a PyTorch dataset from it for training, with no task specified, the
length will be four, as it will correspond to each of the four subjects in the train split. The index
will also cover the full range of each subject's data.

```python
>>> from meds_torchdata.pytorch_dataset import MEDSTorchDataConfig, MEDSPytorchDataset
>>> cfg = MEDSTorchDataConfig(tensorized_cohort_dir=tensorized_MEDS_dataset, max_seq_len=5)
>>> pyd = MEDSPytorchDataset(cfg, split="train")
>>> len(pyd)
4
>>> pyd.index
[(68729, 0, 3), (814703, 0, 3), (239684, 0, 6), (1195293, 0, 8)]
>>> pyd.subject_ids
[68729, 814703, 239684, 1195293]

```

Note the index is in terms of _event indices_, not _measurement indices_ -- meaning it is the index of the
unique timestamp corresponding to the start and end of each subject's data; not the unique measurement. We can
validate that against the raw data. To do so, we'll define the simple helper function `get_event_bounds` that
will just group by the `subject_id` and `time` columns, and then calculate the event index for each subject
and show us the min and max such index, per-subject.

```python
>>> def get_event_indices(df: pl.DataFrame) -> pl.DataFrame:
...     return (
...         df
...         .group_by("subject_id", "time", maintain_order=True).agg(pl.len().alias("n_measurements"))
...         .with_row_index()
...         .select(
...             "subject_id", "time",
...             (pl.col("index") - pl.col("index").min().over("subject_id")).alias("event_idx"),
...             "n_measurements",
...         )
...     )
>>> def get_event_bounds(df: pl.DataFrame) -> pl.DataFrame:
...     return (
...         get_event_indices(df)
...         .with_columns(
...             pl.col("event_idx").max().over("subject_id").alias("max_event_idx")
...         )
...         .filter((pl.col("event_idx") == 0) | (pl.col("event_idx") == pl.col("max_event_idx")))
...         .select("subject_id", "event_idx", "time")
...     )
>>> get_event_bounds(train_1)
shape: (4, 3)
┌────────────┬───────────┬─────────────────────┐
│ subject_id ┆ event_idx ┆ time                │
│ ---        ┆ ---       ┆ ---                 │
│ i64        ┆ u32       ┆ datetime[μs]        │
╞════════════╪═══════════╪═════════════════════╡
│ 68729      ┆ 0         ┆ null                │
│ 68729      ┆ 3         ┆ 2010-05-26 04:51:52 │
│ 814703     ┆ 0         ┆ null                │
│ 814703     ┆ 3         ┆ 2010-02-05 07:02:30 │
└────────────┴───────────┴─────────────────────┘
>>> get_event_bounds(train_0)
shape: (4, 3)
┌────────────┬───────────┬─────────────────────┐
│ subject_id ┆ event_idx ┆ time                │
│ ---        ┆ ---       ┆ ---                 │
│ i64        ┆ u32       ┆ datetime[μs]        │
╞════════════╪═══════════╪═════════════════════╡
│ 239684     ┆ 0         ┆ null                │
│ 239684     ┆ 6         ┆ 2010-05-11 19:27:19 │
│ 1195293    ┆ 0         ┆ null                │
│ 1195293    ┆ 8         ┆ 2010-06-20 20:50:04 │
└────────────┴───────────┴─────────────────────┘

```

While the raw data has codes as strings, naturally, when embedded in the pytorch dataset, they'll get
converted to integers. This happens during the forementioned tensorization step. We can see how the codes are
mapped to integers by looking at the output code metadata of that step:

```python
>>> code_metadata = pl.read_parquet(tensorized_MEDS_dataset.joinpath("metadata/codes.parquet"))
>>> code_metadata.select("code", "code/vocab_index")
shape: (11, 2)
┌───────────────────────┬──────────────────┐
│ code                  ┆ code/vocab_index │
│ ---                   ┆ ---              │
│ str                   ┆ u8               │
╞═══════════════════════╪══════════════════╡
│ ADMISSION//CARDIAC    ┆ 1                │
│ ADMISSION//ORTHOPEDIC ┆ 2                │
│ ADMISSION//PULMONARY  ┆ 3                │
│ DISCHARGE             ┆ 4                │
│ DOB                   ┆ 5                │
│ …                     ┆ …                │
│ EYE_COLOR//BROWN      ┆ 7                │
│ EYE_COLOR//HAZEL      ┆ 8                │
│ HEIGHT                ┆ 9                │
│ HR                    ┆ 10               │
│ TEMP                  ┆ 11               │
└───────────────────────┴──────────────────┘

```

We can see these vocab indices being used if we look at some elements of the pytorch dataset. Note that some
elements of the returned dictionaries are
[`JointNestedRaggedTensorDict`](https://github.com/mmcdermott/nested_ragged_tensors) objects, so we'll define
a helper here that will use a helper from the associated library to help us pretty-print out outputs. Note
that we'll also reduce precision in the numeric values to make the output more readable.

```python
>>> from nested_ragged_tensors.ragged_numpy import pprint_dense
>>> def print_element(el: dict):
...     for k, v in el.items():
...         print(f"{k} ({type(v).__name__}):")
...         if k == "dynamic":
...             pprint_dense(v.to_dense())
...         else:
...             print(v)
>>> print_element(pyd[0])
static_code (list):
[8, 9]
static_numeric_value (list):
[nan, -0.5438239574432373]
dynamic (JointNestedRaggedTensorDict):
code
[ 5  3 10 11  4]
.
numeric_value
[        nan         nan -1.4474752  -0.34049404         nan]
.
time_delta_days
[           nan 1.17661045e+04 0.00000000e+00 0.00000000e+00
 9.78703722e-02]

```

The contents of `pyd[0]` are stable, because index element 0, `(68729, 0, 3)`, indicates the first subject has
a sequence of length 3 in the dataset and our `max_seq_len` is set to 5.

```python
>>> print_element(pyd[0])
static_code (list):
[8, 9]
static_numeric_value (list):
[nan, -0.5438239574432373]
dynamic (JointNestedRaggedTensorDict):
code
[ 5  3 10 11  4]
.
numeric_value
[        nan         nan -1.4474752  -0.34049404         nan]
.
time_delta_days
[           nan 1.17661045e+04 0.00000000e+00 0.00000000e+00
 9.78703722e-02]

```

If we sampled a different subject, one with more than 5 events, the output we'd get would be dependent on the
`config.seq_sampling_strategy` option, and could be non-deterministic. By default, this is set to `random`, so
we'll get a random subset of length 5 each time. Here, so that this code is deterministic, we'll use the
internal seeded version of the getitem call, which just allows to add a seed onto the normal getitem call.

```python
>>> print_element(pyd._seeded_getitem(3, seed=0))
static_code (list):
[6, 9]
static_numeric_value (list):
[nan, 0.06802856922149658]
dynamic (JointNestedRaggedTensorDict):
code
[10 11 10 11 10]
.
numeric_value
[-0.04626633  0.69391906 -0.30007038  0.79735875 -0.31064537]
.
time_delta_days
[0.01888889 0.         0.0084838  0.         0.01167824]
>>> print_element(pyd._seeded_getitem(3, seed=1))
static_code (list):
[6, 9]
static_numeric_value (list):
[nan, 0.06802856922149658]
dynamic (JointNestedRaggedTensorDict):
code
[10 11 10 11 10]
.
numeric_value
[ 0.03833488  0.79735875  0.33972722  0.7456389  -0.04626633]
.
time_delta_days
[0.00115741 0.         0.01373843 0.         0.01888889]

```

We can also examine not just individual elements, but full batches, that we can access with the appropriate
`collate` function via the built in `get_dataloader` method:

```python
>>> print_element(next(iter(pyd.get_dataloader(batch_size=2))))
time_delta_days (Tensor):
tensor([[0.0000e+00, 1.1766e+04, 0.0000e+00, 0.0000e+00, 9.7870e-02],
        [0.0000e+00, 1.2367e+04, 0.0000e+00, 0.0000e+00, 4.6424e-02]])
code (Tensor):
tensor([[ 5,  3, 10, 11,  4],
        [ 5,  2, 10, 11,  4]])
mask (Tensor):
tensor([[True, True, True, True, True],
        [True, True, True, True, True]])
numeric_value (Tensor):
tensor([[ 0.0000,  0.0000, -1.4475, -0.3405,  0.0000],
        [ 0.0000,  0.0000,  3.0047,  0.8491,  0.0000]])
numeric_value_mask (Tensor):
tensor([[False, False,  True,  True, False],
        [False, False,  True,  True, False]])
static_code (Tensor):
tensor([[8, 9],
        [8, 9]])
static_numeric_value (Tensor):
tensor([[ 0.0000, -0.5438],
        [ 0.0000, -1.1012]])
static_numeric_value_mask (Tensor):
tensor([[False,  True],
        [False,  True]])

```

Thus far, our examples have all worked with the default config object, which sets (among other things) the
default output to be at a _measurement_ level, rather than an _event_ level, by virtue of setting
`do_flatten_tensors` to `True`. Let's see what happens if we change that:

```python
>>> pyd.config.do_flatten_tensors = False
>>> print_element(pyd[0])
static_code (list):
[8, 9]
static_numeric_value (list):
[nan, -0.5438239574432373]
dynamic (JointNestedRaggedTensorDict):
time_delta_days
[           nan 1.17661045e+04 9.78703722e-02]
.
---
.
dim1/mask
[[ True False False]
 [ True  True  True]
 [ True False False]]
.
code
[[ 5  0  0]
 [ 3 10 11]
 [ 4  0  0]]
.
numeric_value
[[        nan  0.          0.        ]
 [        nan -1.4474752  -0.34049404]
 [        nan  0.          0.        ]]
>>> print_element(next(iter(pyd.get_dataloader(batch_size=2))))
time_delta_days (Tensor):
tensor([[0.0000e+00, 1.1766e+04, 9.7870e-02],
        [0.0000e+00, 1.2367e+04, 4.6424e-02]])
code (Tensor):
tensor([[[ 5,  0,  0],
         [ 3, 10, 11],
         [ 4,  0,  0]],
<BLANKLINE>
        [[ 5,  0,  0],
         [ 2, 10, 11],
         [ 4,  0,  0]]])
mask (Tensor):
tensor([[True, True, True],
        [True, True, True]])
numeric_value (Tensor):
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.0000, -1.4475, -0.3405],
         [ 0.0000,  0.0000,  0.0000]],
<BLANKLINE>
        [[ 0.0000,  0.0000,  0.0000],
         [ 0.0000,  3.0047,  0.8491],
         [ 0.0000,  0.0000,  0.0000]]])
numeric_value_mask (Tensor):
tensor([[[False,  True,  True],
         [False,  True,  True],
         [False,  True,  True]],
<BLANKLINE>
        [[False,  True,  True],
         [False,  True,  True],
         [False,  True,  True]]])
static_code (Tensor):
tensor([[8, 9],
        [8, 9]])
static_numeric_value (Tensor):
tensor([[ 0.0000, -0.5438],
        [ 0.0000, -1.1012]])
static_numeric_value_mask (Tensor):
tensor([[False,  True],
        [False,  True]])

```

## 📚 Documentation

### Design Principles

A good PyTorch dataset class should:

- Be easy to use
- Have a minimal, constant resource footprint (memory, CPU, start-up time) during model training and
    inference, _regardless of the overall dataset size_.
- Perform as much work as possible in _static, re-usable dataset pre-processing_, rather than upon
    construction or in the __getitem__ method.
- Induce effectively negligible computational overhead in the __getitem__ method relative to model training.
- Be easily configurable, with a simple, consistent API, and cover the most common use-cases.
- Encourage efficient use of GPU resources in the resulting batches.
- Should be comprehensively documented, tested, and benchmarked for performance implications so users can
    use it reliably and effectively.

To achieve this, MEDS TorchData leverages the following design principles:

1. **Lazy Loading**: Data is loaded only when needed, and only the data needed for the current batch is
    loaded.
2. **Efficient Loading**: Data is loaded efficiently leveraging the
    [HuggingFace Safetensors](https://huggingface.co/docs/safetensors/en/index) library for raw IO through
    the nested, ragged interface encoded in the
    [Nested Ragged Tensors](https://github.com/mmcdermott/nested_ragged_tensors) library.
3. **Configurable, Transparent Pre-processing**: Mandatory data pre-processing prior to effective use in
    this library is managed through a simple
    [MEDS-Transforms](https://meds-transforms.readthedocs.io/en/latest/) pipeline which can be run on any
    MEDS dataset, after any model-specific pre-processing, via a transparent configuration file.
4. **Continuous Integration**: The library is continuously tested and benchmarked for performance
    implications, and the results are available to users.

### API and Usage

#### Data Tensorization and Pre-processing

The `MEDS_tensorize` command-line utility is used to convert the input MEDS data into a format that can be
loaded into the PyTorch dataset class contained in this package. This command performs a very simple series of
steps:

1. Normalize the data into an appropriate, numerical format, including:
    \- Assigning each unique `code` in the data a unique integer index and converting the codes to those
    integer indices.
    \- Normalizing the `numeric_value` field to have a mean of 0 and a standard deviation of 1. _If you would
    like additional normalization options supported, such as min-max normalization, please file a GitHub
    issue._
2. Produce a set of static, "schema" files that contain the unique time-points of each subjects' events as
    well as their static measurements.
3. Produce a set of `JointNestedRaggedTensorDict` object files that contain each subjects' dynamic
    measurements in the form of nested, ragged tensors that can be efficiently loaded via the associated
    [package](https://github.com/mmcdermott/nested_ragged_tensors)

These are the only three steps this pipeline performs. Note, however, that this does not mean you can't or
shouldn't perform additional, _model specific pre-processing_ on the data _prior to running the tensorization
command_ for your specific use-case. Indeed, if you wish to perform additional pre-processing, such as

- Dropping numeric values entirely and converting to quantile-modified codes.
- Drop infrequent codes or aggregate codes into higher-order categories.
- Restrict subjects to a specific time-window
- Drop subjects with infrequent values
- Occlude outlier numeric values
- etc.
    You should perform these steps on the raw MEDS data _prior to running the tensorization command_. This ensures
    that the data is modified as you desire in an efficient, transparent way and that the tensorization step works
    with data in its final format to avoid any issues with discrepancies in code vocabulary, etc.

#### Dataset Class

Once the data has been tensorized, you can use the `MEDSPytorchDataset` class to load the data into a PyTorch
dataset suitable to begin modeling! This dataset class takes a configuration object as input, with the
following fields:

## Performance

See https://mmcdermott.github.io/meds-torch-data/dev/bench/ for performance benchmarks for all commits in this
repository. See [here](benchmark/run.py) for the benchmarking script. Note that these benchmarks are likely to
change over time so should be judged relative to the content of the associated commits, not in absolute terms
(e.g., we are likely to benchmark on more or more complex synthetic data, etc.).
