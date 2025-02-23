"""Tests a multi-stage pre-processing pipeline via the Runner utility. Only checks final outputs.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.

In this test, the following stages are run:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization
  - tokenization
  - tensorization

The stage configuration arguments will be as given in the yaml block below:
"""

from functools import partial

from meds import code_metadata_filepath, subject_splits_filepath

RUNNER_SCRIPT = "MEDS_transform-runner"

MEDS_SHARDS = parse_shards_yaml(
    """
train/0: |-2
  subject_id,time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00",DOB,
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
  239684,"05/11/2010, 17:41:51",HR,102.6
  239684,"05/11/2010, 17:41:51",TEMP,96.0
  239684,"05/11/2010, 17:48:48",HR,105.1
  239684,"05/11/2010, 17:48:48",TEMP,96.2
  239684,"05/11/2010, 18:25:35",HR,113.4
  239684,"05/11/2010, 18:25:35",TEMP,95.8
  239684,"05/11/2010, 18:57:18",HR,112.6
  239684,"05/11/2010, 18:57:18",TEMP,95.5
  239684,"05/11/2010, 19:27:19",DISCHARGE,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00",DOB,
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

train/1: |-2
  subject_id,time,code,numeric_value
  68729,,EYE_COLOR//HAZEL,
  68729,,HEIGHT,160.3953106166676
  68729,"03/09/1978, 00:00:00",DOB,
  68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
  68729,"05/26/2010, 02:30:56",HR,86.0
  68729,"05/26/2010, 02:30:56",TEMP,97.8
  68729,"05/26/2010, 04:51:52",DISCHARGE,
  814703,,EYE_COLOR//HAZEL,
  814703,,HEIGHT,156.48559093209357
  814703,"03/28/1976, 00:00:00",DOB,
  814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
  814703,"02/05/2010, 05:55:39",HR,170.2
  814703,"02/05/2010, 05:55:39",TEMP,100.1
  814703,"02/05/2010, 07:02:30",DISCHARGE,

tuning/0: |-2
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

held_out/0: |-2
  subject_id,time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00",DOB,
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
    """
)

MEDS_CODE_METADATA_CSV = """
code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/sum,values/sum_sqd,description,parent_codes
,44,4,28,3198.8389005974336,382968.28937288234,,
ADMISSION//CARDIAC,2,2,0,,,,
ADMISSION//ORTHOPEDIC,1,1,0,,,,
ADMISSION//PULMONARY,1,1,0,,,,
DISCHARGE,4,4,0,,,,
DOB,4,4,0,,,,
EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
"""

MEDS_CODE_METADATA_SCHEMA = {
    "code": pl.Utf8,
    "code/n_occurrences": pl.UInt8,
    "code/n_subjects": pl.UInt8,
    "values/n_occurrences": pl.UInt8,
    "values/n_subjects": pl.UInt8,
    "values/sum": pl.Float32,
    "values/sum_sqd": pl.Float32,
    "values/n_ints": pl.UInt8,
    "values/min": pl.Float32,
    "values/max": pl.Float32,
    "description": pl.Utf8,
    "parent_codes": pl.Utf8,
    "code/vocab_index": pl.UInt8,
}


def parse_code_metadata_csv(csv_str: str) -> pl.DataFrame:
    cols = csv_str.strip().split("\n")[0].split(",")
    schema = {col: dt for col, dt in MEDS_CODE_METADATA_SCHEMA.items() if col in cols}
    df = pl.read_csv(StringIO(csv_str), schema=schema)
    if "parent_codes" in cols:
        df = df.with_columns(pl.col("parent_codes").cast(pl.List(pl.Utf8)))
    return df


MEDS_CODE_METADATA = parse_code_metadata_csv(MEDS_CODE_METADATA_CSV)


def test_pipeline():
    shared_kwargs = {
        "config_name": "runner",
        "stage_name": None,
        "stage_kwargs": None,
        "do_pass_stage_name": False,
        "do_use_config_yaml": False,
        "do_include_dirs": False,
        "hydra_verbose": False,
    }

    single_stage_tester(
        script=str(RUNNER_SCRIPT) + " -h",
        input_files={},
        want_outputs={},
        assert_no_other_outputs=True,
        should_error=False,
        test_name="Runner Help Test",
        stdout_regex=exact_str_regex(NO_ARGS_HELP_STR.strip()),
        **shared_kwargs,
    )

    single_stage_tester(
        script=str(RUNNER_SCRIPT) + " -h",
        input_files={"pipeline.yaml": partial(add_params, PIPELINE_YAML)},
        want_outputs={},
        assert_no_other_outputs=True,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        test_name="Runner Help Test",
        stdout_regex=exact_str_regex(WITH_CONFIG_HELP_STR.strip()),
        **shared_kwargs,
    )

    shared_kwargs["script"] = RUNNER_SCRIPT

    single_stage_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_YAML),
            "stage_runner.yaml": STAGE_RUNNER_YAML,
        },
        want_outputs={
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_VOCABULARY_INDICES,
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
            **WANT_TOKENIZATION_SCHEMAS,
            **WANT_TOKENIZATION_EVENT_SEQS,
            **WANT_NRTs,
        },
        assert_no_other_outputs=False,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        stage_runner_fp="{input_dir}/stage_runner.yaml",
        test_name="Runner Test",
        df_check_kwargs={"check_column_order": False},
        **shared_kwargs,
    )

    single_stage_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_YAML),
            "stage_runner.yaml": PARALLEL_STAGE_RUNNER_YAML,
        },
        want_outputs={
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_VOCABULARY_INDICES,
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
            **WANT_TOKENIZATION_SCHEMAS,
            **WANT_TOKENIZATION_EVENT_SEQS,
            **WANT_NRTs,
        },
        assert_no_other_outputs=False,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        stage_runner_fp="{input_dir}/stage_runner.yaml",
        test_name="Runner Test with parallelism",
        df_check_kwargs={"check_column_order": False},
        **shared_kwargs,
    )

    single_stage_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "_preprocess.yaml": partial(add_params, PIPELINE_YAML),
        },
        should_error=True,
        pipeline_config_fp="{input_dir}/_preprocess.yaml",
        test_name="Runner should fail if the pipeline config has an invalid name",
        **shared_kwargs,
    )

    single_stage_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_NO_STAGES_YAML),
        },
        should_error=True,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        test_name="Runner should fail if the pipeline has no stages",
        **shared_kwargs,
    )
