import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from mixins import SeedableMixin
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .config import SubsequenceSamplingStrategy, SeqPaddingSide, MEDSTorchDataConfig

logger = logging.getLogger(__name__)


BINARY_LABEL_COL = "boolean_value"


class MEDSPytorchDataset(SeedableMixin, torch.utils.data.Dataset):
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
        TODO

    Examples:
        >>> TODO
    """

    @staticmethod
    def subsample_subject_data(
        subject_data: JointNestedRaggedTensorDict,
        max_seq_len: int,
        sampling_strategy: SubsequenceSamplingStrategy,
        do_flatten_tensors: bool = True,
        global_st: int = 0,
    ) -> tuple[JointNestedRaggedTensorDict, int, int]:
        """Subsample subject data based on maximum sequence length and sampling strategy.
    
        This function handles subsampling for both flattened and nested tensor structures.
    
        Args:
            subject_data: Input tensor dictionary containing the sequence data
            max_seq_len: Maximum allowed sequence length
            sampling_strategy: Strategy for selecting subsequence (RANDOM, TO_END, FROM_START)
            do_flatten_tensors: Whether to flatten tensors before subsampling
            global_st: Starting index offset for maintaining global indexing
    
        Returns:
            tuple containing:
            - Subsampled tensor dictionary
            - New global start index
            - New global end index
    
        Examples:
            >>> import numpy as np
            >>> np.random.seed(42)
            >>> # Create sample nested data
            >>> tensors = {
            ...     "code": [[1,2],[3,4],[5,6],[7,8,9,10],[11,12]],
            ...     "time": [0,1,2,3,4],
            ... }
            >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
            >>> # Test FROM_START strategy without flattening
            >>> subsampled, st, end = subsample_subject_data(
            ...     data, max_seq_len=2,
            ...     sampling_strategy=SubsequenceSamplingStrategy.FROM_START,
            ...     do_flatten_tensors=False
            ... )
            >>> subsampled.tensors["dim1/code"].tolist()
            [1, 2, 3, 4]
            >>> subsampled.tensors["dim0/time"].tolist()
            [0, 1]
            >>> st, end
            (0, 2)
    
            >>> # Test TO_END strategy with flattening
            >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
            >>> subsampled, st, end = subsample_subject_data(
            ...     data, max_seq_len=4,
            ...     sampling_strategy=SubsequenceSamplingStrategy.TO_END,
            ...     do_flatten_tensors=True
            ... )
            >>> subsampled.tensors["dim0/code"].tolist()
            [9, 10, 11, 12]
            >>> subsampled.tensors["dim0/time"].tolist()
            [0, 0, 4, 0]
            >>> st, end
            (3, 5)
    
            >>> # Test TO_END strategy
            >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
            >>> subsampled, st, end = subsample_subject_data(
            ...     data, max_seq_len=2,
            ...     sampling_strategy=SubsequenceSamplingStrategy.TO_END,
            ...     do_flatten_tensors=False,
            ... )
            >>> st, end
            (3, 5)
    
            >>> # Test RANDOM strategy
            >>> data = JointNestedRaggedTensorDict(raw_tensors=tensors)
            >>> subsampled, st, end = subsample_subject_data(
            ...     data, max_seq_len=2,
            ...     sampling_strategy=SubsequenceSamplingStrategy.RANDOM,
            ...     do_flatten_tensors=True,
            ... )
            >>> len(subsampled.tensors["dim0/code"]) == 2
            True
        """
        seq_len = len(subject_data)
    
        if do_flatten_tensors:
            # Store original lengths for each time step before flattening
            cum_lens = subject_data.tensors["dim1/bounds"]
    
            subject_data = subject_data.flatten()
            seq_len = len(subject_data)
            if seq_len > max_seq_len:
                match sampling_strategy:
                    case SubsequenceSamplingStrategy.RANDOM:
                        start_offset = np.random.choice(seq_len - max_seq_len)
                    case SubsequenceSamplingStrategy.TO_END:
                        start_offset = seq_len - max_seq_len
                    case SubsequenceSamplingStrategy.FROM_START:
                        start_offset = 0
                    case _:
                        raise ValueError(f"Invalid subsequence sampling strategy {sampling_strategy}!")
            else:
                start_offset = 0
            end = min(seq_len, start_offset + max_seq_len)
            subject_data = subject_data[start_offset:end]
    
            # Map flattened indices back to original time indices
            new_global_st = global_st + np.searchsorted(cum_lens, start_offset, side="right").item()
            new_global_end = global_st + np.searchsorted(cum_lens, end, side="right").item()
        else:
            if seq_len <= max_seq_len:
                return subject_data, global_st, global_st + seq_len
            match sampling_strategy:
                case SubsequenceSamplingStrategy.RANDOM:
                    start_offset = np.random.choice(seq_len - max_seq_len)
                case SubsequenceSamplingStrategy.TO_END:
                    start_offset = seq_len - max_seq_len
                case SubsequenceSamplingStrategy.FROM_START:
                    start_offset = 0
                case _:
                    raise ValueError(f"Invalid subsequence sampling strategy {sampling_strategy}!")
    
            end = min(seq_len, start_offset + max_seq_len)
            subject_data = subject_data[start_offset:end]
    
            new_global_st = global_st + start_offset
            new_global_end = new_global_st + len(subject_data)
    
        return subject_data, new_global_st, new_global_end
    
    
    @staticmethod
    def get_task_indices_and_labels(
        task_df: pl.DataFrame, static_dfs: dict[str, pl.DataFrame]
    ) -> tuple[list[tuple[int, int, int]], dict[str, list]]:
        """Processes the joint DataFrame to determine the index range for each subject's task.
    
        For each row in task_df_joint, it is assumed that `time` is a sorted column and the function
        computes the index of the last event at `prediction_time`.
    
        Parameters:
            - task_df_joint (DataFrame): A DataFrame resulting from the merge_task_with_static function.
    
        Returns:
            - list: list of index tuples of format (subject_id, start_idx, end_idx).
            - dict: dictionary of task names to lists of labels in the same order as the indexes.
    
        Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     shard = "train/shard_0"
        ...     task_df = pl.read_parquet(Path(config.data_dir) / "task_labels.parquet")
        ...     static_dfs = {"shard_0": pl.read_parquet(Path(config.data_dir) / "schema/train/shard_0.parquet")}
        >>> task_df
        shape: (3, 3)
        ┌────────────┬─────────────────────┬───────────────┐
        │ subject_id ┆ prediction_time     ┆ boolean_value │
        │ ---        ┆ ---                 ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ i64           │
        ╞════════════╪═════════════════════╪═══════════════╡
        │ 0          ┆ 1998-01-01 00:00:00 ┆ 0             │
        │ 1          ┆ 1998-01-01 00:00:00 ┆ 1             │
        │ 2          ┆ 1998-01-01 00:00:00 ┆ 0             │
        └────────────┴─────────────────────┴───────────────┘
        >>> static_dfs["shard_0"]
        shape: (3, 5)
        ┌────────────┬─────────────────────┬─────────────────────────────────┬───────────┬─────────────────┐
        │ subject_id ┆ start_time          ┆ time                            ┆ code      ┆ numeric_value   │
        │ ---        ┆ ---                 ┆ ---                             ┆ ---       ┆ ---             │
        │ i64        ┆ datetime[μs]        ┆ list[datetime[μs]]              ┆ list[i64] ┆ list[f64]       │
        ╞════════════╪═════════════════════╪═════════════════════════════════╪═══════════╪═════════════════╡
        │ 0          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
        │ 1          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
        │ 2          ┆ 1995-01-01 00:00:00 ┆ [1995-01-01 00:00:00, 1996-01-… ┆ [1, 2, 3] ┆ [0.1, 0.2, 0.3] │
        └────────────┴─────────────────────┴─────────────────────────────────┴───────────┴─────────────────┘
        >>>
        >>> # Run the function
        >>> BINARY_LABEL_COL = "boolean_value"  # Define the constant used in the function
        >>> indices, labels, pred_times = get_task_indices_and_labels(task_df, static_dfs)
        >>>
        >>> # Check the results
        >>> print(indices)  # Only subjects 1 and 2 should be present (inner join)
        [(0, 4), (1, 4), (2, 4)]
        >>> print(labels)  # Labels for subjects 1 and 2
        [0, 1, 0]
        """
    
        static_df = pl.concat(static_dfs.values()).select("subject_id", "time")
    
        end_idx_expr = (
            (pl.col("time").search_sorted(pl.col("prediction_time"), side="right")).last().alias("end_idx")
        )
    
        label_df = (
            task_df.join(static_df, on="subject_id", how="inner")
            .with_row_index("_row_index")
            .explode("time")
            .group_by("_row_index", "subject_id", "prediction_time", "boolean_value", maintain_order=True)
            .agg(end_idx_expr)
        )
    
        indexes = list(zip(label_df["subject_id"], label_df["end_idx"]))
        labels = label_df[BINARY_LABEL_COL].to_list()
        prediction_times = label_df["prediction_time"].to_list()
    
        return indexes, labels, prediction_times

    def __init__(self, cfg: MEDSTorchDataConfig, split: str):
        super().__init__()

        self.config = cfg
        self.split = split

        logger.info("Reading subject schema and static data")
        self.read_subject_descriptors()

    def read_subject_descriptors(self):
        """Read subject schemas and static data from the dataset.

        This method processes the Parquet files for each shard in the dataset, extracting static data and
        creating various mappings and indices for efficient data access.

        The method populates the following instance attributes:
        - self.static_dfs: Dictionary of static DataFrames for each shard.
        - self.subj_indices: Mapping of subject IDs to their indices.
        - self.subj_seq_bounds: Dictionary of sequence bounds for each subject.
        - self.index: List of (subject_id, start, end) tuples for data access.
        - self.labels: Dictionary of task labels (if tasks are specified).

        If a task is specified in the configuration, this method also processes the task labels and
        integrates them with the static data.

        Raises:
            ValueError: If duplicate subjects are found across shards.
            FileNotFoundError: If specified task files are not found.
        """

        schema_root = Path(self.config.schema_files_root)

        self.static_dfs = {}
        self.subj_indices = {}
        self.subj_seq_bounds = {}
        self.subj_map = {}

        schema_files = list(schema_root.glob(f"{self.split}/*.parquet"))
        if not schema_files:
            raise FileNotFoundError(
                f"No schema files found in {schema_root}! If your data is not sharded by split, this error "
                "may occur because this codebase does not handle non-split sharded data. See Issue #79 for "
                "tracking this issue."
            )

        for schema_fp in schema_files:
            shard = str(schema_fp.relative_to(schema_root).with_suffix(""))

            df = (
                pl.read_parquet(
                    schema_fp,
                    columns=[
                        "subject_id",
                        "start_time",
                        "code",
                        "numeric_value",
                        "time",
                    ],
                    use_pyarrow=True,
                )
                .rename({"code": "static_indices", "numeric_value": "static_values"})
                .with_columns(
                    pl.col("static_values").list.eval(pl.element().fill_null(0)),
                    pl.col("static_indices").list.eval(pl.element().fill_null(0)),
                )
            )

            self.static_dfs[shard] = df
            subject_ids = df["subject_id"]
            self.subj_map.update({subj: shard for subj in subject_ids})

            n_events = df.select(pl.col("time").list.len().alias("n_events")).get_column("n_events")
            for i, (subj, n_events_count) in enumerate(zip(subject_ids, n_events)):
                if subj in self.subj_indices or subj in self.subj_seq_bounds:
                    raise ValueError(f"Duplicate subject {subj} in {shard}!")

                self.subj_indices[subj] = i
                self.subj_seq_bounds[subj] = (0, n_events_count)

        if self.has_task:
            task_df_fp = Path(self.config.task_label_path)
            if not task_df_fp.is_file():
                logger.info(f"If the task file is not found at {task_df_fp}")
                task_df_fp = task_df_fp.with_suffix("") / "**/*.parquet"
                logger.info(f"Searching for task parquets over the glob {task_df_fp}")

            logger.info(f"Reading task constraints for {self.config.task_name} from {task_df_fp}")
            task_df = pl.read_parquet(task_df_fp)

            subjs_and_ends, self.labels, self.prediction_times = get_task_indices_and_labels(
                task_df, self.static_dfs
            )
            self.index = [(subj, 0, end) for subj, end in subjs_and_ends]
        else:
            self.index = [(subj, *bounds) for subj, bounds in self.subj_seq_bounds.items()]
            self.labels = None

    @property
    def subject_ids(self) -> list[int]:
        return [x[0] for x in self.index]

    def __len__(self):
        return len(self.index)

    @property
    def has_task(self) -> bool:
        return self.config.task_name is not None

    @property
    def max_seq_len(self) -> int:
        return self.config.max_seq_len

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
                - static_indices(Optional): List of static MEDS codes.
                - static_values(Optional): List of static MEDS numeric values.
                - static_mask(Optional): List of static masks (True means the value is static).

        """

        subject_dynamic_data, subject_id, st, end = self.load_subject_dynamic_data(idx)

        out = self.load_subject(subject_dynamic_data, subject_id, st, end)

        if self.config.do_include_subject_id:
            out["subject_id"] = subject_id
        if self.config.do_include_prediction_time:
            if not self.has_task:
                if not self.config.do_include_end_time:
                    raise ValueError(
                        "Cannot include prediction_time without a task specified or do_include_end_time!"
                    )
                else:
                    out["prediction_time"] = out["end_time"]
            else:
                out["prediction_time"] = self.prediction_times[idx]

        if self.labels is not None:
            out[BINARY_LABEL_COL] = self.labels[idx]

        return out

    def load_subject_dynamic_data(self, idx: int):
        """Loads and returns the dynamic data slice for a given subject index, with subject ID and time range.

        Args:
            idx (int): Index of the subject in the dataset index

        Returns:
            tuple: (subject_dynamic_data, subject_id, st, end) where:
                - subject_dynamic_data is a JointNestedRaggedTensorDict containing the dynamic data
                - subject_id is the ID of the subject
                - st is the start time index
                - end is the end time index

        Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

        >>> # Create a test dataset and initialize the PytorchDataset
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = PytorchDataset(config, split='train')
        ...
        ...     # Test loading dynamic data for first subject
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...     print(f"Subject ID: {subject_id}")
        ...     print(f"Time range: {st} to {end}")
        ...     print(f"Dynamic data keys: {sorted(dynamic_data.tensors.keys())}")
        Subject ID: 0
        Time range: 0 to 4
        Dynamic data keys: ['dim0/time_delta_days', 'dim1/bounds', 'dim1/code', 'dim1/numeric_value']

        >>> # Test loading dynamic data for second subject
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = PytorchDataset(config, split='train')
        ...
        ...     # Load second subject
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(1)
        ...     print(f"Subject ID: {subject_id}")
        ...     print(f"Time range: {st} to {end}")
        ...     # Verify data structure
        ...     print(f"Has numeric values: {'dim1/numeric_value' in dynamic_data.tensors}")
        ...     print(f"Has time deltas: {'dim0/time_delta_days' in dynamic_data.tensors}")
        Subject ID: 1
        Time range: 0 to 4
        Has numeric values: True
        Has time deltas: True

        >>> # Test error case with invalid index
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = PytorchDataset(config, split='train')
        ...     try:
        ...         dynamic_data = dataset.load_subject_dynamic_data(999)  # Invalid index
        ...     except IndexError as e:
        ...         print("Caught expected IndexError")
        Caught expected IndexError
        """
        subject_id, st, end = self.index[idx]
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]

        dynamic_data_fp = Path(self.config.data_dir) / "data" / f"{shard}.nrt"

        subject_dynamic_data = JointNestedRaggedTensorDict(tensors_fp=dynamic_data_fp)[subject_idx, st:end]

        return subject_dynamic_data, subject_id, st, end

    @SeedableMixin.WithSeed
    def load_subject(
        self, subject_dynamic_data, subject_id: int, global_st: int, global_end: int
    ) -> dict[str, list[float]]:
        """Load and process data for a single subject.

        Args:
            subject_dynamic_data: The dynamic data for the subject.
            subject_id (int): The ID of the subject to load.
            global_st (int): The start index of the sequence to load.
            global_end (int): The end index of the sequence to load.

        Returns:
            dict: A dictionary containing the processed data for the subject.

        Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> import numpy as np
        >>> from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

        >>> # Test basic subject loading
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     dataset = PytorchDataset(config, split='train')
        ...
        ...     # First get the dynamic data using load_subject_dynamic_data
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...
        ...     # Then load the complete subject data
        ...     subject_data = dataset.load_subject(dynamic_data, subject_id, st, end)
        ...
        ...     # Verify the returned data structure
        ...     print("Keys in subject data:")
        ...     for key in sorted(subject_data.keys()): print(f"{key}")
        ...     print()
        ...     print(f"Has static indices: {len(subject_data['static_indices']) > 0}")
        ...     print(f"Has dynamic data: {isinstance(subject_data['dynamic'], JointNestedRaggedTensorDict)}")
        ...     print(f"Has end time: {'end_time' in subject_data}")
        Keys in subject data:
        dynamic
        end_idx
        end_time
        start_idx
        start_time
        static_indices
        static_values
        <BLANKLINE>
        Has static indices: True
        Has dynamic data: True
        Has end time: True

        >>> # Test with different configuration settings
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     # Create config with modified settings
        ...     config = create_dummy_dataset(tmp_dir)
        ...     config.do_prepend_static_data = False
        ...     config.postpend_eos_token = False
        ...     config.do_include_start_time_min = False
        ...
        ...     dataset = PytorchDataset(config, split='train')
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...     subject_data = dataset.load_subject(dynamic_data, subject_id, st, end)
        ...
        ...     # Verify the modified behavior
        ...     print(f"Contains start time: {'start_time' in subject_data}")
        ...     dynamic_tensors = subject_data['dynamic'].tensors
        ...     has_eos = np.any(dynamic_tensors['dim0/code'] == config.EOS_TOKEN_ID)
        ...     print(f"Contains EOS token: {has_eos}")
        Contains start time: False
        Contains EOS token: False

        >>> # Test with maximum sequence length constraint
        >>> with tempfile.TemporaryDirectory() as tmp_dir:
        ...     config = create_dummy_dataset(tmp_dir)
        ...     config.max_seq_len = 5  # Set small max sequence length
        ...
        ...     dataset = PytorchDataset(config, split='train')
        ...     dynamic_data, subject_id, st, end = dataset.load_subject_dynamic_data(0)
        ...     subject_data = dataset.load_subject(dynamic_data, subject_id, st, end)
        ...
        ...     # Verify sequence length constraints
        ...     dynamic_len = len(subject_data['dynamic'].tensors['dim0/code'])
        ...     print(f"Dynamic sequence length: {dynamic_len}")
        ...     print(f"Respects max length: {dynamic_len <= config.max_seq_len}")
        Dynamic sequence length: 5
        Respects max length: True
        """
        shard = self.subj_map[subject_id]
        subject_idx = self.subj_indices[subject_id]
        static_row = self.static_dfs[shard][subject_idx].to_dict()

        max_seq_len = self.config.max_seq_len

        out = {
            "static_indices": static_row["static_indices"].item().to_list(),
            "static_values": static_row["static_values"].item().to_list(),
        }

        if self.config.do_prepend_static_data:
            n_static = len(out["static_indices"])
            if n_static >= max_seq_len:
                raise ValueError(
                    f"Static data length {n_static} matches or exceeds "
                    f"max_seq_len {max_seq_len} for subject {subject_id}!"
                )

            max_seq_len -= n_static
        if self.config.postpend_eos_token:
            max_seq_len -= 1

        subject_dynamic_data, global_st, global_end = subsample_subject_data(
            subject_dynamic_data,
            max_seq_len,
            self.config.subsequence_sampling_strategy,
            self.config.do_flatten_tensors,
            global_st,
        )

        if self.config.do_include_subsequence_indices:
            out["start_idx"] = global_st
            out["end_idx"] = global_end

        tensors = subject_dynamic_data.tensors

        if self.config.do_prepend_static_data:
            tensors["dim0/time_delta_days"] = np.concatenate(
                [np.zeros(len(out["static_indices"])), tensors["dim0/time_delta_days"]]
            )
            tensors["dim0/static_mask"] = np.concatenate(
                [
                    np.ones(len(out["static_indices"]), dtype=bool),
                    np.zeros(len(tensors["dim0/code"]), dtype=bool),
                ]
            )
            tensors["dim0/code"] = np.concatenate([out["static_indices"], tensors["dim0/code"]])
            tensors["dim0/numeric_value"] = np.concatenate(
                [out["static_values"], tensors["dim0/numeric_value"]]
            )
        else:
            tensors["dim0/static_mask"] = np.zeros(len(tensors["dim0/code"]), dtype=bool)

        if self.config.postpend_eos_token:
            tensors["dim0/code"] = np.append(tensors["dim0/code"], [self.config.EOS_TOKEN_ID])
            tensors["dim0/static_mask"] = np.append(tensors["dim0/static_mask"], [False])
            tensors["dim0/numeric_value"] = np.append(tensors["dim0/numeric_value"], [0])
            tensors["dim0/time_delta_days"] = np.append(tensors["dim0/time_delta_days"], [0])

        subject_dynamic_data = JointNestedRaggedTensorDict(processed_tensors=tensors)

        out["dynamic"] = subject_dynamic_data

        if self.config.do_include_start_time_min:
            out["start_time"] = static_row["time"].item().to_list()[global_st]
        if self.config.do_include_end_time:
            out["end_time"] = static_row["time"].item().to_list()[global_end - 1]

        return out


    def collate(self, batch: list[dict]) -> dict:
        """Combines a batch of data points into a single, tensorized batch.

        The collated output is a fully tensorized and padded dictionary, ready for input into an
        `input_encoder`. This method uses the JointNestedRaggedTensorDict API to collate and pad the data.

        Args:
            batch (list[dict]): A list of dictionaries, each representing a single sample as
                returned by the __getitem__ method.

        Returns:
            dict: A dictionary containing the collated batch data.
        """

        data = JointNestedRaggedTensorDict.vstack([item["dynamic"] for item in batch]).to_dense()
        tensorized = {k: torch.as_tensor(v) for k, v in data.items()}
        tensorized["code"] = tensorized["code"].long()
        tensorized["mask"] = tensorized.pop("dim1/mask")
        tensorized["numeric_value_mask"] = ~torch.isnan(tensorized["numeric_value"])
        tensorized["time_delta_days"] = torch.nan_to_num(tensorized["time_delta_days"], nan=0).float()
        tensorized["numeric_value"] = torch.nan_to_num(tensorized["numeric_value"], nan=0).float()

        # Add task labels to batch
        for k in batch[0].keys():
            if k not in ("dynamic", "static_values", "static_indices", "static_mask"):
                if isinstance(batch[0][k], datetime):
                    tensorized[k] = [item[k] for item in batch]
                else:
                    tensorized[k] = torch.Tensor([item[k] for item in batch])
        return tensorized
