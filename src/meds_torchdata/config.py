"""Contains configuration objects for building a PyTorch dataset from a MEDS dataset.

This module contains configuration objects for building a PyTorch dataset from a MEDS dataset. These include
enumeration objects for categorical options and a general DataClass configuration object for dataset options.
"""

from dataclasses import dataclass
from enum import StrEnum

from numpy.random import Generator, default_rng


class SubsequenceSamplingStrategy(StrEnum):
    """An enumeration of the possible subsequence sampling strategies for the dataset.

    Attributes:
        RANDOM: Randomly sample a subsequence from the full sequence.
        TO_END: Sample a subsequence from the end of the full sequence.
            Note this starts at the last element and moves back.
        FROM_START: Sample a subsequence from the start of the full sequence.

    Methods:
        subsample_st_offset: Subsample starting offset based on maximum sequence length and sampling strategy.
            This method can be used on instances
            (e.g., SubsequenceSamplingStrategy.RANDOM.subsample_st_offset) but is most often used as a static
            class level method for maximal clarity.
    """

    RANDOM = "random"
    TO_END = "to_end"
    FROM_START = "from_start"

    def subsample_st_offset(
        strategy,
        seq_len: int,
        max_seq_len: int,
        rng: Generator | None = None,
    ) -> int | None:
        """Subsample starting offset based on maximum sequence length and sampling strategy.

        Args:
            strategy: Strategy for selecting subsequence (RANDOM, TO_END, FROM_START)
            seq_len: Length of the sequence
            max_seq_len: Maximum allowed sequence length
            rng: Random number generator for random sampling. If None, a new generator is created.

        Returns:
            The (integral) start offset within the sequence based on the sampling strategy, or `None` if no
            subsampling is required.

        Examples:
            >>> SubsequenceSamplingStrategy.subsample_st_offset("from_start", 10, 5)
            0
            >>> SubsequenceSamplingStrategy.subsample_st_offset(SubsequenceSamplingStrategy.TO_END, 10, 5)
            5
            >>> SubsequenceSamplingStrategy.subsample_st_offset("random", 10, 5, rng=default_rng(1))
            2
            >>> SubsequenceSamplingStrategy.RANDOM.subsample_st_offset(10, 10) is None
            True
            >>> SubsequenceSamplingStrategy.subsample_st_offset("foo", 10, 5)
            Traceback (most recent call last):
                ...
            ValueError: Invalid subsequence sampling strategy foo!
        """

        if seq_len <= max_seq_len:
            return None

        match strategy:
            case SubsequenceSamplingStrategy.RANDOM:
                if rng is None:
                    rng = default_rng()
                return rng.choice(seq_len - max_seq_len)
            case SubsequenceSamplingStrategy.TO_END:
                return seq_len - max_seq_len
            case SubsequenceSamplingStrategy.FROM_START:
                return 0
            case _:
                raise ValueError(f"Invalid subsequence sampling strategy {strategy}!")


class SeqPaddingSide(StrEnum):
    """An enumeration of the possible sequence padding sides for the dataset.

    Attributes:
        LEFT: Pad the sequences on the left side (e.g., [[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]]).
        RIGHT: Pad the sequences on the right side (e.g., [[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]]).
    """

    LEFT = "left"
    RIGHT = "right"


@dataclass
class MEDSTorchDataConfig:
    """A data class for storing configuration options for building a PyTorch dataset from a MEDS dataset.

    Attributes:
        seq_len: The maximum length of sequences to yield from the dataset.
        seq_padding_side: The side to pad the sequences on.
        subseq_sampling_strategy: The subsequence sampling strategy for the dataset.
        subseq_len: The length of the subsequences in the dataset.
    """

    # MEDS Dataset Information
    MEDS_dataset_dir: str

    # Sequence lengths and padding
    max_seq_len: int
    seq_padding_side: SeqPaddingSide
    seq_sampling_strategy: SubsequenceSamplingStrategy

    # Static Data
    do_prepend_static_data: bool

    # Task Labels
    task_labels_dir: str

    # Output Shape & Masking
    do_flatten_tensors: bool = True
    do_include_event_mask: bool = True

    # Include Metadata in batches
    do_include_subject_id: bool = False
    do_include_subsequence_indices: bool = False
    do_include_start_time: bool = False
    do_include_end_time: bool = False
    do_include_prediction_time: bool = False
