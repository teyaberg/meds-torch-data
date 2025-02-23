"""Tests the full, multi-stage pre-processing pipeline. Only checks tokenized and tensorized outputs."""


from meds import code_metadata_filepath, subject_splits_filepath

from . import (
    MEDS_CODE_METADATA,
    MEDS_SHARDS,
    PREPROCESS_SCRIPT,
    SPLITS_DF,
    single_stage_tester,
)
from .test_tensorization import WANT_NRTS
from .test_tokenization import WANT_EVENT_SEQS, WANT_SCHEMAS


def test_preprocess():
    single_stage_tester(
        script=str(PREPROCESS_SCRIPT),
        stage_name=None,
        stage_kwargs=None,
        do_pass_stage_name=False,
        do_use_config_yaml=False,
        do_include_dirs=False,
        hydra_verbose=False,
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
        },
        want_outputs={
            **WANT_SCHEMAS,
            **WANT_EVENT_SEQS,
            **WANT_NRTS,
        },
        assert_no_other_outputs=False,
        should_error=False,
        test_name="Pre-process Test",
        df_check_kwargs={"check_column_order": False},
    )
