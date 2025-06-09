from bisect import bisect_right
from datetime import datetime, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from meds import DataSchema, LabelSchema

from meds_torchdata import MEDSPytorchDataset


def _schema_and_labels():
    """Strategy generating a schema DataFrame and a corresponding label DataFrame."""

    @st.composite
    def _strategy(draw):
        n_subjects = draw(st.integers(min_value=1, max_value=4))
        subject_ids = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=n_subjects, max_size=n_subjects, unique=True
            )
        )
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 10)
        schema_times = []
        for _ in subject_ids:
            n_times = draw(st.integers(min_value=1, max_value=5))
            times = draw(
                st.lists(st.datetimes(min_value=start, max_value=end), min_size=n_times, max_size=n_times)
            )
            schema_times.append(sorted(times))
        schema_df = pl.DataFrame(
            {DataSchema.subject_id_name: subject_ids, DataSchema.time_name: schema_times}
        )

        n_labels = draw(st.integers(min_value=1, max_value=6))
        label_rows = []
        for _ in range(n_labels):
            subj = draw(st.one_of(st.sampled_from(subject_ids), st.integers(min_value=50, max_value=60)))
            pred_time = draw(
                st.datetimes(min_value=start - timedelta(days=1), max_value=end + timedelta(days=1))
            )
            value = draw(st.booleans())
            label_rows.append(
                {
                    DataSchema.subject_id_name: subj,
                    LabelSchema.prediction_time_name: pred_time,
                    LabelSchema.boolean_value_name: value,
                }
            )
        label_df = pl.DataFrame(label_rows)
        return schema_df, label_df

    return _strategy()


@given(_schema_and_labels())
@settings(max_examples=25)
def test_get_task_seq_bounds_and_labels_property(data):
    schema_df, label_df = data

    result = MEDSPytorchDataset.get_task_seq_bounds_and_labels(label_df, schema_df)

    # Drop labels for subjects not in schema_df
    label_subset = label_df.filter(
        pl.col(DataSchema.subject_id_name).is_in(schema_df[DataSchema.subject_id_name])
    )

    expected_rows = []
    schema_map = {
        row[0]: row[1]
        for row in schema_df.select(DataSchema.subject_id_name, DataSchema.time_name).iter_rows()
    }
    for row in label_subset.iter_rows(named=True):
        times = schema_map[row[DataSchema.subject_id_name]]
        idx = bisect_right(times, row[LabelSchema.prediction_time_name])
        expected_rows.append(
            {
                DataSchema.subject_id_name: row[DataSchema.subject_id_name],
                MEDSPytorchDataset.END_IDX: idx,
                LabelSchema.prediction_time_name: row[LabelSchema.prediction_time_name],
                LabelSchema.boolean_value_name: row[LabelSchema.boolean_value_name],
            }
        )

    expected = pl.DataFrame(expected_rows)

    assert result.to_dict(False) == expected.to_dict(False)


@given(st.data())
@settings(max_examples=25)
def test_schema_df_last_observed(sample_dataset_config_with_index, data):
    cfg = sample_dataset_config_with_index
    cfg.include_window_last_observed_in_schema = True
    dataset = MEDSPytorchDataset(cfg, split="train")

    idx = data.draw(st.integers(min_value=0, max_value=len(dataset) - 1))
    subj, end_idx = dataset.index[idx]
    shard, subj_idx = dataset.subj_locations[subj]
    times = dataset.schema_dfs_by_shard[shard][DataSchema.time_name][subj_idx]

    assert 0 < end_idx <= len(times)
    assert dataset.schema_df[dataset.LAST_TIME][idx] == times[end_idx - 1]
