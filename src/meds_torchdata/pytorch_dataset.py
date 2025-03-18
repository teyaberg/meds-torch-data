import logging
from typing import NamedTuple

import numpy as np
import polars as pl
import torch
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .config import MEDSTorchDataConfig, StaticInclusionMode

logger = logging.getLogger(__name__)


BINARY_LABEL_COL = "boolean_value"


class StaticData(NamedTuple):
    code: list[int]
    numeric_value: list[float | None]


class MEDSPytorchDataset(torch.utils.data.Dataset):
    """A PyTorch dataset that provides efficient PyTorch access to a MEDS dataset.

    This dataset is designed to work with data from the MEDS (Medical Event Data Set) format, supporting
    various types of medical events, static patient information, and task-specific labels. It provides
    functionality for loading, processing, and collating data for use in PyTorch models in an efficient manner
    that takes advantage of the sparsity of EHR data to minimize memory usage and computational time.

    Key design principles:
      1. The class will store an `index` variable that specifies what is the valid range of data to consider
         for any given subject in the dataset corresponding to an integer index passed to `__getitem__`.
      2. Data will only be loaded for subjects on an as-needed basis, and will not be cached, to minimize
         memory usage during normal operation.
      3. As much work as possible should be relegated to separate dataset pre-processing (resulting in files
         stored on disk) rather than this class to streamline operation.
      4. The primary input to this class in terms of data is a pre-processed set of "schema files" and "nested
         ragged tensor" data files that can be used to identify the shape of the dataset and to efficiently
         load the relevant tensor data, respectively.

    Args:
        cfg: Configuration options for the dataset, realized through a dataclass instance.
        split: The data split to use. This must match up to the splits stored in the root dataset's
               `metadata/subject_splits.parquet` file's `split` column.

    Attributes:
        config: The configuration options for the dataset.
        split: The data split to use.
        schema_dfs: A dictionary of static DataFrames for each shard.
        subj_to_row_idx: A mapping of subject IDs to their indices.
        index: A list of (subject_id, start, end) tuples for data access.
        labels: A dictionary of task labels (if tasks are specified).

    For examples of this class, see the global README.md. Here, we'll include some examples of other aspects
    of the class, such as error validation and specific methods.

    Examples:

        If you load the dataset from real data, things work fine.

        >>> cfg = MEDSTorchDataConfig(tensorized_cohort_dir=tensorized_MEDS_dataset, max_seq_len=5)
        >>> pyd = MEDSPytorchDataset(cfg, split="train")
        >>> len(pyd)
        4
        >>> pyd.index
        [(68729, 0, 3), (814703, 0, 3), (239684, 0, 6), (1195293, 0, 8)]

        If you pass in a non-existent split, you'll get an error as it won't be able to find the schema files:

        >>> pyd = MEDSPytorchDataset(cfg, split="nonexistent") # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        FileNotFoundError: No schema files found in /tmp/.../tokenization/schemas! If your data is not sharded
        by split, this error may occur because this codebase does not handle non-split sharded data. See Issue
        #79 for tracking this issue.

    """

    @staticmethod
    def get_task_seq_bounds_and_labels(
        label_df: pl.DataFrame, schema_df: pl.DataFrame
    ) -> tuple[list[tuple[int, int]], list[bool]]:
        """Returns the event-level allowed input sequence boundaries and labels for each task sample.

        This function is guaranteed to output an index of the same order and length as `label_df`. Subjects
        not present in `schema_df` will be included in the output, with null labels and indices.

        Args:
            label_df: The DataFrame containing the task labels, in the MEDS Label DF schema.
            schema_df: A DataFrame with subject ID and a list of event timestamps for each shard.

        Returns:
            A list of pairs of subject IDs and the event index one larger than the last permissible event for
            that subject, and a list of labels for the subject in the same order.

        Examples:
            >>> label_df = pl.DataFrame({
            ...     "subject_id": [1, 2, 2, 4, 3, 3, 3],
            ...     "prediction_time": [
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 1), datetime(2020, 1, 2),
            ...         datetime(2020, 1, 1),
            ...         datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3),
            ...     ],
            ...     "boolean_value": [True, False, True, False, True, False, True],
            ... })
            >>> schema_df = pl.DataFrame({
            ...     "subject_id": [2, 6, 1, 3],
            ...     "time": [
            ...         # Subject 2: Prediction times are 2020-1-1,2020-1-2
            ...         [
            ...             datetime(2019, 12, 31),
            ...             datetime(2019, 12, 31, 12),
            ...             datetime(2019, 12, 31, 23, 59, 59),
            ...             datetime(2020, 1, 1, 0, 0, 1),
            ...             datetime(2020, 1, 2),
            ...             datetime(2020, 1, 20),
            ...         ],
            ...         # Subject 6: No prediction times
            ...         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            ...         # Subject 1: Prediction times are 2020-1-1
            ...         [datetime(2019, 12, 1), datetime(2020, 1, 1), datetime(2020, 1, 2)],
            ...         # Subject 3: Prediction times are 2020-1-1,2020-1-2,2020-1-3
            ...         [datetime(2020, 1, 1), datetime(2021, 11, 2), datetime(2021, 11, 3)],
            ...     ],
            ... })
            >>> subjs_ends, labels = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)
            >>> subjs_ends
            [(1, 2), (2, 3), (2, 5), (3, 1), (3, 1), (3, 1)]
            >>> labels
            [True, False, True, True, False, True]
        """

        end_idx_expr = (
            (pl.col("time").search_sorted(pl.col("prediction_time"), side="right")).last().alias("end_idx")
        )

        label_df = (
            label_df.join(schema_df, on="subject_id", how="inner", maintain_order="left")
            .with_row_index("_row_index")
            .explode("time")
            .group_by("_row_index", "subject_id", "prediction_time", "boolean_value", maintain_order=True)
            .agg(end_idx_expr)
        )

        indexes = list(zip(label_df["subject_id"], label_df["end_idx"]))
        labels = label_df[BINARY_LABEL_COL].to_list()

        return indexes, labels

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Reading subject schema and static data")

        self.schema_dfs = {}
        self.subj_to_row_idx = {}
        self.subj_to_shard = {}

        subj_seq_bounds = {}
        for shard, schema_fp in self.config.schema_fps:
            if not shard.startswith(f"{self.split}/"):
                continue

            df = pl.read_parquet(schema_fp, use_pyarrow=True).with_columns(
                pl.col("static_code").list.eval(pl.element().fill_null(0)),
                pl.col("static_numeric_value").list.eval(pl.element().fill_null(np.nan)),
            )

            self.schema_dfs[shard] = df
            subject_ids = df["subject_id"]
            self.subj_to_shard.update({subj: shard for subj in subject_ids})

            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(subject_ids, n_events)):
                self.subj_to_row_idx[subj] = i
                subj_seq_bounds[subj] = (0, n_events_count)

        if not self.schema_dfs:
            raise FileNotFoundError(
                f"No schema files found in {self.config.schema_dir}! If your data is not sharded by split, "
                "this error may occur because this codebase does not handle non-split sharded data. See "
                "Issue #79 for tracking this issue."
            )

        if self.has_task:
            logger.info(f"Reading tasks from {self.config.task_labels_fps}")

            schema_df = pl.concat(df.select("subject_id", "time") for df in self.schema_dfs.values())

            subjs_and_ends, self.labels = self.get_task_seq_bounds_and_labels(self.labels_df, schema_df)
            self.index = [(subj, 0, end) for subj, end in subjs_and_ends]
        else:
            self.index = [(subj, *bounds) for subj, bounds in subj_seq_bounds.items()]
            self.labels = None

    @property
    def labels_df(self) -> pl.DataFrame:
        """Returns the task labels as a DataFrame, in the MEDS Label schema, or `None` if there is no task.

        Examples:
            >>> print(sample_pytorch_dataset.labels_df)
            None
            >>> sample_pytorch_dataset_with_task.labels_df.select(
            ...     "subject_id", "prediction_time", "boolean_value"
            ... )
            shape: (21, 3)
            ┌────────────┬─────────────────────┬───────────────┐
            │ subject_id ┆ prediction_time     ┆ boolean_value │
            │ ---        ┆ ---                 ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ bool          │
            ╞════════════╪═════════════════════╪═══════════════╡
            │ 239684     ┆ 2010-05-11 18:00:00 ┆ false         │
            │ 239684     ┆ 2010-05-11 18:30:00 ┆ true          │
            │ 239684     ┆ 2010-05-11 19:00:00 ┆ true          │
            │ 1195293    ┆ 2010-06-20 19:30:00 ┆ false         │
            │ 1195293    ┆ 2010-06-20 20:00:00 ┆ true          │
            │ …          ┆ …                   ┆ …             │
            │ 754281     ┆ 2010-01-03 08:00:00 ┆ true          │
            │ 1500733    ┆ 2010-06-03 15:00:00 ┆ false         │
            │ 1500733    ┆ 2010-06-03 15:30:00 ┆ false         │
            │ 1500733    ┆ 2010-06-03 16:00:00 ┆ true          │
            │ 1500733    ┆ 2010-06-03 16:30:00 ┆ true          │
            └────────────┴─────────────────────┴───────────────┘
        """
        if not self.has_task:
            return None

        return pl.concat(
            [pl.read_parquet(fp, use_pyarrow=True) for fp in self.config.task_labels_fps], how="vertical"
        )

    @property
    def subject_ids(self) -> list[int]:
        """Returns the list of subject IDs for whom data is returned for each index. May have duplicates.

        Examples:
            >>> sample_pytorch_dataset.subject_ids
            [68729, 814703, 239684, 1195293]
            >>> sample_pytorch_dataset_with_task.subject_ids # doctest: +NORMALIZE_WHITESPACE
            [239684, 239684, 239684,
             1195293, 1195293, 1195293,
             68729, 68729, 68729, 68729,
             814703, 814703, 814703]
        """
        return [x[0] for x in self.index]

    def __len__(self):
        """Returns the length of the dataset.

        Examples:
            >>> len(sample_pytorch_dataset)
            4
            >>> len(sample_pytorch_dataset_with_task)
            13
        """
        return len(self.index)

    @property
    def has_task(self) -> bool:
        """Returns whether the dataset has a task specified. A convenience wrapper around the config property.

        Examples:
            >>> sample_pytorch_dataset.has_task
            False
            >>> sample_pytorch_dataset_with_task.has_task
            True
        """
        return self.config.has_task

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset.

        This method returns a dictionary corresponding to a single subject's data at the specified index. The
        data is not tensorized in this method, as that work is typically done in the collate function.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the data for the specified index. The structure typically includes:
                - code: List of categorical metadata elements.
                - mask: Mask of valid elements in the sequence, False means it is a padded element.
                - numeric_value: List of dynamic numeric values.
                - numeric_value_mask: Mask of numeric values (False means no numeric value was recorded)
                - time_delta_days: List of dynamic time deltas between observations.
                - static_code (Optional): List of static MEDS codes.
                - static_numeric_value (Optional): List of static MEDS numeric values.
        """
        return self._seeded_getitem(idx)

    def _seeded_getitem(self, idx: int, seed: int | None = None) -> dict[str, torch.Tensor]:
        """Retrieve a single data point from the dataset with a specified random seed.

        This is merely a deterministic wrapper around the `_getitem` method that allows for deterministic
        subsequence sampling.
        """

        dynamic_data, static_data = self.load_subject_data(*self.index[idx])

        out = {"static_code": static_data.code, "static_numeric_value": static_data.numeric_value}

        out["dynamic"] = self.config.process_dynamic_data(dynamic_data, rng=seed)

        if self.has_task:
            out[BINARY_LABEL_COL] = self.labels[idx]

        return out

    def load_subject_data(
        self, subject_id: int, st: int | None, end: int | None
    ) -> tuple[JointNestedRaggedTensorDict, StaticData]:
        """Loads and returns the dynamic data slice for a given subject ID and permissible event range.

        Args:
            subject_id: The ID of the subject to load.
            st: The (integral) index of the first permissible event (meaning unique timestamp) that can be
                read for this subject's record. If None, no limit is applied.
            end: The (integral) index of the last permissible event (meaning unique timestamp) that can be
                 read for this subject's record. If None, no limit is applied.

        Returns:
            The subject's dynamic data and static data. The static data is returned as a StaticData named
            tuple with two fields: `code` and `numeric_value`.

        Examples:
            >>> from nested_ragged_tensors.ragged_numpy import pprint_dense
            >>> dynamic_data, static_data = sample_pytorch_dataset.load_subject_data(68729, 0, 3)
            >>> static_data.code
            [8, 9]
            >>> static_data.numeric_value
            [nan, -0.5438239574432373]
            >>> pprint_dense(dynamic_data.to_dense())
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

            To see that these make sense, recall we can check the raw data. Obviously, the data have been
            normalized and tokenized, so we should not expect exact matches in the numeric values or code
            strings, but were we to inspect the code vocabularies, they would align:

            >>> from meds_testing_helpers.dataset import MEDSDataset
            >>> D = MEDSDataset(root_dir=simple_static_MEDS)
            >>> raw_data = pl.from_arrow(D.data_shards["train/1"]).filter(pl.col("subject_id") == 68729)
            >>> raw_data
            shape: (7, 4)
            ┌────────────┬─────────────────────┬──────────────────────┬───────────────┐
            │ subject_id ┆ time                ┆ code                 ┆ numeric_value │
            │ ---        ┆ ---                 ┆ ---                  ┆ ---           │
            │ i64        ┆ datetime[μs]        ┆ str                  ┆ f32           │
            ╞════════════╪═════════════════════╪══════════════════════╪═══════════════╡
            │ 68729      ┆ null                ┆ EYE_COLOR//HAZEL     ┆ null          │
            │ 68729      ┆ null                ┆ HEIGHT               ┆ 160.395309    │
            │ 68729      ┆ 1978-03-09 00:00:00 ┆ DOB                  ┆ null          │
            │ 68729      ┆ 2010-05-26 02:30:56 ┆ ADMISSION//PULMONARY ┆ null          │
            │ 68729      ┆ 2010-05-26 02:30:56 ┆ HR                   ┆ 86.0          │
            │ 68729      ┆ 2010-05-26 02:30:56 ┆ TEMP                 ┆ 97.800003     │
            │ 68729      ┆ 2010-05-26 04:51:52 ┆ DISCHARGE            ┆ null          │
            └────────────┴─────────────────────┴──────────────────────┴───────────────┘
            >>> subj_codes = raw_data["code"].unique().to_list()
            >>> code_metadata = (
            ...     pl.read_parquet(tensorized_MEDS_dataset / "metadata/codes.parquet")
            ...     .filter(pl.col("code").is_in(subj_codes))
            ... )
            >>> mean_col = (pl.col("values/sum")/pl.col("values/n_occurrences")).alias("values/mean")
            >>> std_col = (
            ...     (pl.col("values/sum_sqd")/pl.col("values/n_occurrences") - mean_col**2)**0.5
            ... ).alias("values/std")
            >>> code_metadata.select("code", "code/vocab_index", mean_col, std_col)
            shape: (7, 4)
            ┌──────────────────────┬──────────────────┬─────────────┬────────────┐
            │ code                 ┆ code/vocab_index ┆ values/mean ┆ values/std │
            │ ---                  ┆ ---              ┆ ---         ┆ ---        │
            │ str                  ┆ u8               ┆ f32         ┆ f32        │
            ╞══════════════════════╪══════════════════╪═════════════╪════════════╡
            │ ADMISSION//PULMONARY ┆ 3                ┆ NaN         ┆ NaN        │
            │ DISCHARGE            ┆ 4                ┆ NaN         ┆ NaN        │
            │ DOB                  ┆ 5                ┆ NaN         ┆ NaN        │
            │ EYE_COLOR//HAZEL     ┆ 8                ┆ NaN         ┆ NaN        │
            │ HEIGHT               ┆ 9                ┆ 164.209732  ┆ 7.014076   │
            │ HR                   ┆ 10               ┆ 113.375     ┆ 18.912241  │
            │ TEMP                 ┆ 11               ┆ 98.458336   ┆ 1.933464   │
            └──────────────────────┴──────────────────┴─────────────┴────────────┘

            Note this is independent of the task data and the index; this only depends on the raw data on
            disk. So, we'll see the exact same output if we call over the sample dataset with tasks because
            the raw MEDS data is the same.

            >>> dynamic_data, static_data = sample_pytorch_dataset_with_task.load_subject_data(68729, 0, 3)
            >>> static_data.code
            [8, 9]
            >>> static_data.numeric_value
            [nan, -0.5438239574432373]
            >>> pprint_dense(dynamic_data.to_dense())
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
        """
        shard = self.subj_to_shard[subject_id]
        subject_idx = self.subj_to_row_idx[subject_id]

        dynamic_data_fp = self.config.tensorized_cohort_dir / "data" / f"{shard}.nrt"
        subject_dynamic_data = JointNestedRaggedTensorDict(tensors_fp=dynamic_data_fp)[subject_idx, st:end]

        subj_schema = self.schema_dfs[shard][subject_idx].to_dict()
        static_code = subj_schema["static_code"].item().to_list()
        static_numeric_value = subj_schema["static_numeric_value"].item().to_list()

        return subject_dynamic_data, StaticData(static_code, static_numeric_value)

    def collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            dict: A dictionary containing the collated batch data.

        Examples:
            >>> batch = [sample_pytorch_dataset[0], sample_pytorch_dataset[1]]
            >>> sample_pytorch_dataset.collate(batch) # doctest: +NORMALIZE_WHITESPACE
            {'time_delta_days': tensor([[0.0000e+00, 1.1766e+04, 0.0000e+00, 0.0000e+00, 9.7870e-02],
                                        [0.0000e+00, 1.2367e+04, 0.0000e+00, 0.0000e+00, 4.6424e-02]]),
             'code': tensor([[ 5,  3, 10, 11,  4],
                             [ 5,  2, 10, 11,  4]]),
             'mask': tensor([[True, True, True, True, True],
                             [True, True, True, True, True]]),
             'numeric_value': tensor([[ 0.0000,  0.0000, -1.4475, -0.3405,  0.0000],
                                      [ 0.0000,  0.0000,  3.0047,  0.8491,  0.0000]]),
             'numeric_value_mask': tensor([[False, False,  True,  True, False],
                                           [False, False,  True,  True, False]]),
             'static_code': tensor([[8, 9],
                                    [8, 9]]),
             'static_numeric_value': tensor([[ 0.0000, -0.5438],
                                             [ 0.0000, -1.1012]]),
             'static_numeric_value_mask': tensor([[False,  True],
                                                  [False,  True]])}
            >>> batch = [sample_pytorch_dataset_with_task[0], sample_pytorch_dataset_with_task[1]]
            >>> sample_pytorch_dataset_with_task.collate(batch) # doctest: +NORMALIZE_WHITESPACE
            {'time_delta_days': tensor([[0.0000e+00, 1.0727e+04, 0.0000e+00, 0.0000e+00, 4.8264e-03,
                                         0.0000e+00, 0.0000e+00, 0.0000e+00],
                                        [0.0000e+00, 1.0727e+04, 0.0000e+00, 0.0000e+00, 4.8264e-03,
                                         0.0000e+00, 2.5544e-02, 0.0000e+00]]),
             'code': tensor([[ 5,  1, 10, 11, 10, 11,  0,  0],
                             [ 5,  1, 10, 11, 10, 11, 10, 11]]),
             'mask': tensor([[ True,  True,  True,  True,  True,  True, False, False],
                             [ True,  True,  True,  True,  True,  True,  True,  True]]),
             'numeric_value': tensor([[ 0.0000e+00,  0.0000e+00, -5.6974e-01, -1.2715e+00, -4.3755e-01,
                                       -1.1680e+00,  0.0000e+00,  0.0000e+00],
                                      [ 0.0000e+00,  0.0000e+00, -5.6974e-01, -1.2715e+00, -4.3755e-01,
                                        -1.1680e+00,  1.3220e-03, -1.3749e+00]]),
             'numeric_value_mask': tensor([[False, False,  True,  True,  True,  True,  True,  True],
                                           [False, False,  True,  True,  True,  True,  True,  True]]),
             'static_code': tensor([[7, 9],
                                    [7, 9]]),
             'static_numeric_value': tensor([[0.0000, 1.5770],
                                             [0.0000, 1.5770]]),
             'static_numeric_value_mask': tensor([[False,  True],
                                                  [False,  True]]),
             'boolean_value': tensor([False,  True])}

            Static data can also be omitted if set in the config.

            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.OMIT
            >>> batch = [sample_pytorch_dataset[0], sample_pytorch_dataset[1]]
            >>> sample_pytorch_dataset.collate(batch) # doctest: +NORMALIZE_WHITESPACE
            {'time_delta_days': tensor([[0.0000e+00, 1.1766e+04, 0.0000e+00, 0.0000e+00, 9.7870e-02],
                                        [0.0000e+00, 1.2367e+04, 0.0000e+00, 0.0000e+00, 4.6424e-02]]),
             'code': tensor([[ 5,  3, 10, 11,  4],
                             [ 5,  2, 10, 11,  4]]),
             'mask': tensor([[True, True, True, True, True],
                             [True, True, True, True, True]]),
             'numeric_value': tensor([[ 0.0000,  0.0000, -1.4475, -0.3405,  0.0000],
                                      [ 0.0000,  0.0000,  3.0047,  0.8491,  0.0000]]),
             'numeric_value_mask': tensor([[False, False,  True,  True, False],
                                           [False, False,  True,  True, False]])}

            >>> sample_pytorch_dataset.config.static_inclusion_mode = StaticInclusionMode.INCLUDE
        """

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}

        out = {}
        out["time_delta_days"] = torch.nan_to_num(tensorized.pop("time_delta_days"), nan=0).float()
        out["code"] = tensorized.pop("code").long()
        out["mask"] = tensorized.pop("dim1/mask")
        out["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()
        out["numeric_value_mask"] = ~torch.isnan(tensorized.pop("numeric_value"))

        match self.config.static_inclusion_mode:
            case StaticInclusionMode.OMIT:
                pass
            case StaticInclusionMode.INCLUDE:
                static_data = JointNestedRaggedTensorDict(
                    {
                        "static_code": [item["static_code"] for item in batch],
                        "static_numeric_value": [item["static_numeric_value"] for item in batch],
                    }
                ).to_dense()
                static_tensorized = {k: torch.as_tensor(v) for k, v in static_data.items()}
                out["static_code"] = static_tensorized.pop("static_code").long()
                out["static_numeric_value"] = torch.nan_to_num(
                    static_tensorized["static_numeric_value"], nan=0
                ).float()
                out["static_numeric_value_mask"] = ~torch.isnan(static_tensorized["static_numeric_value"])

        if self.has_task:
            out[BINARY_LABEL_COL] = torch.Tensor([item[BINARY_LABEL_COL] for item in batch]).bool()

        return out

    def get_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        """Constructs a PyTorch DataLoader for this dataset using the dataset's custom collate function.

        Args:
            **kwargs: Additional arguments to pass to the DataLoader constructor.

        Returns:
            torch.utils.data.DataLoader: A DataLoader object for this dataset.

        Examples:
            >>> DL = sample_pytorch_dataset.get_dataloader(batch_size=2, shuffle=False)
            >>> next(iter(DL)) # doctest: +NORMALIZE_WHITESPACE
            {'time_delta_days': tensor([[0.0000e+00, 1.1766e+04, 0.0000e+00, 0.0000e+00, 9.7870e-02],
                                        [0.0000e+00, 1.2367e+04, 0.0000e+00, 0.0000e+00, 4.6424e-02]]),
             'code': tensor([[ 5,  3, 10, 11,  4],
                             [ 5,  2, 10, 11,  4]]),
             'mask': tensor([[True, True, True, True, True],
                             [True, True, True, True, True]]),
             'numeric_value': tensor([[ 0.0000,  0.0000, -1.4475, -0.3405,  0.0000],
                                      [ 0.0000,  0.0000,  3.0047,  0.8491,  0.0000]]),
             'numeric_value_mask': tensor([[False, False,  True,  True, False],
                                           [False, False,  True,  True, False]]),
             'static_code': tensor([[8, 9],
                                    [8, 9]]),
             'static_numeric_value': tensor([[ 0.0000, -0.5438],
                                             [ 0.0000, -1.1012]]),
             'static_numeric_value_mask': tensor([[False,  True],
                                                  [False,  True]])}
        """
        return torch.utils.data.DataLoader(self, collate_fn=self.collate, **kwargs)
