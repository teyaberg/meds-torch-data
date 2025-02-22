"""Contains configuration objects for building a PyTorch dataset from a MEDS dataset.

This module contains configuration objects for building a PyTorch dataset from a MEDS dataset. These include
enumeration objects for categorical options and a general DataClass configuration object for dataset options. 
"""

from dataclasses import dataclass
from enum import StrEnum


BINARY_LABEL_COL = "boolean_value"


class SubsequenceSamplingStrategy(StrEnum):
    """An enumeration of the possible subsequence sampling strategies for the dataset.

    Attributes:
        RANDOM: Randomly sample a subsequence from the full sequence.
        TO_END: Sample a subsequence from the end of the full sequence.
            Note this starts at the last element and moves back.
        FROM_START: Sample a subsequence from the start of the full sequence.
    """

    RANDOM = "random"
    TO_END = "to_end"
    FROM_START = "from_start"


class SeqPaddingSide(StrEnum):
    """An enumeration of the possible sequence padding sides for the dataset."""

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

    # Include Metadata in batches
    do_include_subject_id: bool = False
    do_include_subsequence_indices: bool = False
    do_include_start_time: bool = False
    do_include_end_time: bool = False
    do_include_prediction_time: bool = False

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
