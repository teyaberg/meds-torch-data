"""Contains configuration objects for building a PyTorch dataset from a MEDS dataset.

This module contains configuration objects for building a PyTorch dataset from a MEDS dataset. These include
enumeration objects for categorical options and a general DataClass configuration object for dataset options.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .types import BatchMode, StaticInclusionMode, SubsequenceSamplingStrategy

logger = logging.getLogger(__name__)


@dataclass
class MEDSTorchDataConfig:
    """A data class for storing configuration options for building a PyTorch dataset from a MEDS dataset.

    Attributes:
        max_seq_len: The maximum length of sequences to yield from the dataset.
        seq_sampling_strategy: The subsequence sampling strategy for the dataset.

    Raises:
        FileNotFoundError: If the task_labels_dir or the tensorized_cohort_dir is not a valid directory.
        ValueError: If the subsequence sampling strategy or static inclusion mode is not valid.
        ValueError: If the task_labels_dir is specified but the subsequence sampling strategy is not TO_END.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir: # No error
        ...     cfg = MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=tmpdir,
        ...         max_seq_len=10,
        ...     )
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_root = Path(tmpdir) / "tensorized"
        ...     cohort_root.mkdir()
        ...     task_labels_dir = Path(tmpdir) / "task_labels"
        ...     task_labels_dir.mkdir()
        ...     cfg = MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=cohort_root,
        ...         max_seq_len=10,
        ...         task_labels_dir=task_labels_dir,
        ...         seq_sampling_strategy="to_end",
        ...     )

        If the cohort directory doesn't exist, an error is raised.

        >>> with tempfile.TemporaryDirectory() as tmpdir: # Error as cohort dir doesn't exist
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=Path(tmpdir) / "non_existent",
        ...         max_seq_len=10,
        ...     )
        Traceback (most recent call last):
            ...
        FileNotFoundError: tensorized_cohort_dir must be a valid directory. Got ...

        If the task labels directory doesn't exist, an error is raised.

        >>> with tempfile.TemporaryDirectory() as tmpdir: # Error as task labels dir doesn't exist
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=tmpdir,
        ...         max_seq_len=10,
        ...         task_labels_dir=Path(tmpdir) / "non_existent",
        ...     )
        Traceback (most recent call last):
            ...
        FileNotFoundError: If specified, task_labels_dir must be a valid directory. Got ...

        If the subsequence sampling strategy is not TO_END when a task is specified an error is raised.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_root = Path(tmpdir) / "tensorized"
        ...     cohort_root.mkdir()
        ...     task_labels_dir = Path(tmpdir) / "task_labels"
        ...     task_labels_dir.mkdir()
        ...     MEDSTorchDataConfig(
        ...         tensorized_cohort_dir=cohort_root,
        ...         max_seq_len=10,
        ...         task_labels_dir=task_labels_dir,
        ...         seq_sampling_strategy="random",
        ...     )
        Traceback (most recent call last):
            ...
        ValueError: Not sampling data till the end of the sequence when predicting for a specific task is not
        permitted! This is because there is no use-case we know of where you would want to do this. If you
        disagree, please let us know via a GitHub issue.

        If the subsequence sampling strategy or static inclusion mode is not valid, an error is raised.

        >>> MEDSTorchDataConfig(tensorized_cohort_dir=".", max_seq_len=3, seq_sampling_strategy="foobar")
        Traceback (most recent call last):
            ...
        ValueError: Invalid subsequence sampling strategy: foobar
        >>> MEDSTorchDataConfig(tensorized_cohort_dir=".", max_seq_len=3, static_inclusion_mode="foobar")
        Traceback (most recent call last):
            ...
        ValueError: Invalid static inclusion mode: foobar
    """

    # MEDS Dataset Information
    tensorized_cohort_dir: str

    # Sequence lengths and padding
    max_seq_len: int
    seq_sampling_strategy: SubsequenceSamplingStrategy = SubsequenceSamplingStrategy.RANDOM

    # Static Data
    static_inclusion_mode: StaticInclusionMode = StaticInclusionMode.INCLUDE

    # Task Labels
    task_labels_dir: str | None = None

    # Output Shape & Masking
    batch_mode: BatchMode = BatchMode.SM

    def __post_init__(self):
        self.tensorized_cohort_dir = Path(self.tensorized_cohort_dir)
        if not self.tensorized_cohort_dir.is_dir():
            raise FileNotFoundError(
                "tensorized_cohort_dir must be a valid directory. "
                f"Got {self.tensorized_cohort_dir.resolve()!s}"
            )

        match self.static_inclusion_mode:
            case str() if self.static_inclusion_mode in {x.value for x in StaticInclusionMode}:
                self.static_inclusion_mode = StaticInclusionMode(self.static_inclusion_mode)
            case StaticInclusionMode():  # pragma: no cover
                pass
            case _:
                raise ValueError(f"Invalid static inclusion mode: {self.static_inclusion_mode}")

        match self.seq_sampling_strategy:
            case str() if self.seq_sampling_strategy in {x.value for x in SubsequenceSamplingStrategy}:
                self.seq_sampling_strategy = SubsequenceSamplingStrategy(self.seq_sampling_strategy)
            case SubsequenceSamplingStrategy():  # pragma: no cover
                pass
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy: {self.seq_sampling_strategy}")

        if self.task_labels_dir is not None:
            self.task_labels_dir = Path(self.task_labels_dir)
            if not self.task_labels_dir.is_dir():
                raise FileNotFoundError(
                    "If specified, task_labels_dir must be a valid directory. "
                    f"Got {self.task_labels_dir.resolve()!s}"
                )
            if self.seq_sampling_strategy != SubsequenceSamplingStrategy.TO_END:
                raise ValueError(
                    "Not sampling data till the end of the sequence when predicting for a specific task is "
                    "not permitted! This is because there is no use-case we know of where you would want to "
                    "do this. If you disagree, please let us know via a GitHub issue."
                )

    @property
    def schema_dir(self) -> Path:
        """Return the schema directory for the tensorized cohort. The path need not exist to be returned.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     cfg = MEDSTorchDataConfig(Path(tmpdir), max_seq_len=10)
            >>> cfg.schema_dir
            PosixPath('/tmp/tmp.../tokenization/schemas')
        """
        return self.tensorized_cohort_dir / "tokenization" / "schemas"

    @property
    def schema_fps(self) -> Generator[tuple[str, Path], None, None]:
        """Yield shard names and schema paths for existent schema files.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     tensorized_root = Path(tmpdir)
            ...     schema_dir = tensorized_root / "tokenization" / "schemas"
            ...     schema_dir.mkdir(parents=True)
            ...     (schema_dir / "shard_A.parquet").touch()
            ...     (schema_dir / "shard_B.json").touch()
            ...     (schema_dir / "shard_C/").mkdir()
            ...     (schema_dir / "shard_C" / "0.parquet").touch()
            ...     (schema_dir / "shard_C" / "1.parquet").touch()
            ...     (schema_dir / "shard_D/").mkdir()
            ...     cfg = MEDSTorchDataConfig(tensorized_root, max_seq_len=10)
            ...     for shard, fp in cfg.schema_fps:
            ...         print(shard, str(fp.relative_to(tensorized_root)))
            shard_A tokenization/schemas/shard_A.parquet
            shard_C/1 tokenization/schemas/shard_C/1.parquet
            shard_C/0 tokenization/schemas/shard_C/0.parquet
        """

        for schema_fp in self.schema_dir.rglob("*.parquet"):
            shard = str(schema_fp.relative_to(self.schema_dir).with_suffix(""))
            yield shard, schema_fp

    @property
    def task_labels_fps(self) -> list[Path] | None:
        """Returns the list of task label files for this configuration, or `None` if no task is specified.

        Returned files must exist; if no such files exist, will return an empty list.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     tensorized_root = Path(tmpdir) / "tensorized"
            ...     tensorized_root.mkdir()
            ...     cfg_no_task = MEDSTorchDataConfig(tensorized_root, 2)
            ...     print(f"No task dir: {cfg_no_task.task_labels_fps}")
            ...     task_labels_dir = Path(tmpdir) / "task_labels"
            ...     task_labels_dir.mkdir()
            ...     (task_labels_dir / "labels_1.parquet").touch()
            ...     (task_labels_dir / "nested").mkdir()
            ...     (task_labels_dir / "nested/labels_2.parquet").touch()
            ...     cfg_task = MEDSTorchDataConfig(
            ...         tensorized_root, 2, task_labels_dir=task_labels_dir, seq_sampling_strategy="to_end"
            ...     )
            ...     print(f"Task dir: {cfg_task.task_labels_fps}")
            No task dir: None
            Task dir: [PosixPath('/tmp/.../task_labels/labels_1.parquet'),
                       PosixPath('/tmp/.../task_labels/nested/labels_2.parquet')]
        """

        return list(self.task_labels_dir.rglob("*.parquet")) if self.task_labels_dir else None

    def process_dynamic_data(
        self,
        data: JointNestedRaggedTensorDict,
        rng: np.random.Generator | int | None = None,
    ) -> JointNestedRaggedTensorDict:
        """This processes the dynamic data for a subject, including subsampling and flattening.

        Args:
            data: The dynamic data for the subject.
            rng: The random seed to use for subsequence sampling. If `None`, the default rng is used. If an
                integer, a new rng is created with that seed.

        Returns:
            The processed dynamic data, still in a `JointNestedRaggedTensorDict` format.

        Examples:
            >>> from nested_ragged_tensors.ragged_numpy import pprint_dense
            >>> data = JointNestedRaggedTensorDict({
            ...     "time_delta": [1, 2, 3, 4, 5, 6, 7],
            ...     "code": [[10, 11], [20, 21], [30], [40], [50, 51, 52], [60], [70, 71, 72, 73]],
            ... })

            If the config says to sample until the end, we'll just grab the last three elements.

            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="to_end", batch_mode="SEM"
            ... )
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            time_delta
            [5 6 7]
            .
            ---
            .
            dim1/mask
            [[ True  True  True False]
             [ True False False False]
             [ True  True  True  True]]
            .
            code
            [[50 51 52  0]
             [60  0  0  0]
             [70 71 72 73]]

            If we flatten the tensors, then we get only 1D tensors for both, and the time elements that are
            added to account for the longer length are imputed to zero. Note we've increased the `max_seq_len`
            to 5 to show some non-imputed time-deltas.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=5, seq_sampling_strategy="to_end")
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            code
            [60 70 71 72 73]
            .
            time_delta
            [6 7 0 0 0]

            If we sample from the start, we'll just grab the first three elements.

            >>> cfg = MEDSTorchDataConfig(
            ...     ".", max_seq_len=3, seq_sampling_strategy="from_start", batch_mode="SEM"
            ... )
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            time_delta
            [1 2 3]
            .
            ---
            .
            dim1/mask
            [[ True  True]
             [ True  True]
             [ True False]]
            .
            code
            [[10 11]
             [20 21]
             [30  0]]

            Again, if we flatten the tensors, we get only 1D tensors for both.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=3, seq_sampling_strategy="from_start")
            >>> pprint_dense(cfg.process_dynamic_data(data).to_dense())
            code
            [10 11 20]
            .
            time_delta
            [1 0 2]

            Random sampling is non-deterministic, but can be fixed by a seed.

            >>> cfg = MEDSTorchDataConfig(".", max_seq_len=3, seq_sampling_strategy="random")
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=1).to_dense())
            code
            [40 50 51]
            .
            time_delta
            [4 5 0]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=1).to_dense())
            code
            [40 50 51]
            .
            time_delta
            [4 5 0]
            >>> pprint_dense(cfg.process_dynamic_data(data, rng=3).to_dense())
            code
            [52 60 70]
            .
            time_delta
            [0 6 7]
        """

        if self.batch_mode == BatchMode.SM:
            data = data.flatten()

        seq_len = len(data)
        st = self.seq_sampling_strategy.subsample_st_offset(seq_len, self.max_seq_len, rng=rng)

        if st is None:
            st = 0

        end = min(seq_len, st + self.max_seq_len)
        return data[st:end]
