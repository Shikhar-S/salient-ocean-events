"""Unit tests for the BirdSet download *parsing* helpers (no network / no `datasets`).

`download.py` imports `datasets` lazily inside `download()`, so the pure helpers below
import and run without the heavy dep installed.
"""

from src.provenance.birdset.download import (
    event_rows_from_example,
    hf_config_name,
    make_code_decoder,
    master_id_from_filepath,
)


def test_hf_config_name_defaults_to_scape():
    # default skips the focal `train` by loading the soundscape-only `<region>_scape` config
    assert hf_config_name("HSN", True) == "HSN_scape"
    assert hf_config_name("PER", True) == "PER_scape"
    assert hf_config_name("HSN", False) == "HSN"          # full config (focal train included)
    assert hf_config_name("HSN_scape", True) == "HSN_scape"  # no double suffix


def test_master_id_is_sanitized_basename():
    # BirdSet filepath is an absolute extracted path; key on the (unique) recording stem.
    assert master_id_from_filepath(
        "/cache/extracted/HSN_test_shard_0001/HSN_038_20150710_082105.ogg"
    ) == "HSN_038_20150710_082105"
    assert master_id_from_filepath("HSN/2020/PER_001 file.ogg") == "PER_001_file"
    assert master_id_from_filepath("/abs/path/rec.flac") == "rec"
    assert master_id_from_filepath("plain") == "plain"


def test_event_rows_single_code():
    ex = {"start_time": 1.0, "end_time": 2.5, "ebird_code": "amerob"}
    assert list(event_rows_from_example(ex)) == [(1.0, 2.5, "amerob")]


def test_event_rows_multilabel_expands():
    ex = {"start_time": 0.0, "end_time": 1.0, "ebird_code": None,
          "ebird_code_multilabel": ["amerob", "comyel"]}
    rows = list(event_rows_from_example(ex))
    assert rows == [(0.0, 1.0, "amerob"), (0.0, 1.0, "comyel")]


def test_event_rows_skips_nocall_and_bad_spans():
    assert list(event_rows_from_example(
        {"start_time": 1.0, "end_time": 2.0, "ebird_code": "nocall"})) == []
    # missing / non-positive spans are skipped
    assert list(event_rows_from_example(
        {"start_time": None, "end_time": 2.0, "ebird_code": "x"})) == []
    assert list(event_rows_from_example(
        {"start_time": 2.0, "end_time": 2.0, "ebird_code": "x"})) == []
    assert list(event_rows_from_example(
        {"start_time": 3.0, "end_time": 2.0, "ebird_code": "x"})) == []


class _FakeClassLabel:
    """Stand-in for datasets.ClassLabel (has int2str), no `datasets` import needed."""
    def __init__(self, names):
        self.names = names

    def int2str(self, i):
        return self.names[i]


def test_decoder_resolves_classlabel_ints_and_filters():
    # BirdSet's ebird_code is a ClassLabel -> the raw field is an int index.
    feats = {"ebird_code": _FakeClassLabel(["amerob", "comyel", "nocall"])}
    dec = make_code_decoder(feats)
    assert list(event_rows_from_example(
        {"start_time": 0.0, "end_time": 1.0, "ebird_code": 1}, dec)) == [(0.0, 1.0, "comyel")]
    # an int mapping to 'nocall' is dropped
    assert list(event_rows_from_example(
        {"start_time": 0.0, "end_time": 1.0, "ebird_code": 2}, dec)) == []
    # out-of-range / sentinel negative index is dropped, not crashed
    assert list(event_rows_from_example(
        {"start_time": 0.0, "end_time": 1.0, "ebird_code": -1}, dec)) == []
