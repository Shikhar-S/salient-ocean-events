"""Tests for the GT-free annotator: grid scoring + top-K selection on synthetic audio."""

import numpy as np
import soundfile as sf
from pytest import approx

from src.annotate import (
    grid_windows,
    read_wav_scp,
    score_grid,
    select_segments,
    write_segments,
)


# ------------------------------------------------------------------ grid_windows
def test_grid_windows_are_consecutive_fixed_length():
    win = grid_windows(duration_s=35.0, window_s=10.0)
    assert win == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]  # <10 s tail dropped


def test_grid_windows_short_file_is_whole_file():
    assert grid_windows(duration_s=4.0, window_s=10.0) == [(0.0, 4.0)]


# ------------------------------------------------------------------- select
def _row(seg, wid, start, n, mx, sm):
    return {"seg_id": seg, "wav_id": wid, "start": start, "end": start + 10.0,
            "n_events": n, "max_score": mx, "sum_score": sm}


def test_select_drops_empty_and_keeps_top_fraction():
    rows = [
        _row("r-0", "r", 0.0, 0, 0.0, 0.0),    # empty -> always dropped
        _row("r-1", "r", 10.0, 1, 1.0, 1.0),
        _row("r-2", "r", 20.0, 3, 9.0, 20.0),  # most salient
        _row("r-3", "r", 30.0, 2, 5.0, 8.0),
    ]
    kept = select_segments(rows, keep_frac=0.5, rank_by="sum_score", scope="global")
    # 3 non-empty -> ceil(0.5*3)=2 kept, the two highest sum_score
    assert [r["seg_id"] for r in kept] == ["r-2", "r-3"]


def test_select_per_recording_keeps_each_recording():
    rows = [
        _row("a-0", "a", 0.0, 5, 9.0, 50.0),
        _row("a-1", "a", 10.0, 1, 1.0, 1.0),
        _row("b-0", "b", 0.0, 1, 0.5, 0.5),   # quiet recording still represented
        _row("b-1", "b", 10.0, 1, 0.4, 0.4),
    ]
    kept = select_segments(rows, keep_frac=0.5, scope="per_recording")
    wavs = {r["wav_id"] for r in kept}
    assert wavs == {"a", "b"}                 # both recordings kept
    assert [r["seg_id"] for r in kept] == ["a-0", "b-0"]  # top of each, sorted


def test_select_returns_sorted_by_recording_and_start():
    rows = [_row(f"a-{i}", "a", t, 2, 1.0, float(s))
            for i, (t, s) in enumerate([(20.0, 9), (0.0, 8), (10.0, 7)])]
    kept = select_segments(rows, keep_frac=1.0)
    assert [r["start"] for r in kept] == [0.0, 10.0, 20.0]


# ------------------------------------------------------------------- read_wav_scp
def test_read_wav_scp_keeps_spaces_in_path(tmp_path):
    scp = tmp_path / "wav.scp"
    scp.write_text("id1 /a/b/70017ch2 44kHz.wav\nid2 /a/c.wav\n")
    rows = read_wav_scp(str(scp))
    assert rows == [("id1", "/a/b/70017ch2 44kHz.wav"), ("id2", "/a/c.wav")]


# ------------------------------------------------------------------- end to end
def _make_recording(path, sr=48000, dur=35.0, bursts=((3.0, 3.4), (23.0, 23.6))):
    """Silence with white-noise bursts: windows [0,10) and [20,30) are eventful;
    [10,20) is silent."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    rng = np.random.default_rng(0)
    for on, off in bursts:
        a, b = int(on * sr), int(off * sr)
        x[a:b] = 0.6 * rng.standard_normal(b - a).astype(np.float32)
    sf.write(path, x, sr)


def _read_segments(path):
    rows = []
    with open(path) as f:
        for line in f:
            seg_id, wid, start, end = line.split()
            rows.append((seg_id, wid, float(start), float(end)))
    return rows


def test_score_grid_then_select_end_to_end(tmp_path):
    wav = tmp_path / "rec1.wav"
    _make_recording(str(wav))
    scp = tmp_path / "wav.scp"
    scp.write_text(f"rec1 {wav}\n")
    grid = tmp_path / "grid.tsv"
    score_grid(str(scp), str(grid), target_sr=48000, window_s=10.0)

    # one row per 10 s grid window (3 for a 35 s file: tail dropped); header present
    lines = grid.read_text().splitlines()
    assert lines[0].startswith("seg_id\twav_id\tstart\tend")
    assert len(lines) == 1 + 3

    seg = tmp_path / "segments"
    kept = select_segments(str(grid), keep_frac=0.5, scope="per_recording")
    write_segments(kept, str(seg))

    rows = _read_segments(str(seg))
    assert len(rows) >= 1
    for seg_id, wid, start, end in rows:
        assert wid == "rec1" and seg_id.startswith("rec1-")
        assert end - start == approx(10.0, abs=0.01)
    # the silent [10,20) window must not be among the most-salient kept ones
    assert all(not (start == approx(10.0)) for _, _, start, _ in rows)


def test_sharding_partitions_recordings_disjointly(tmp_path):
    scp = tmp_path / "wav.scp"
    lines = []
    for i in range(4):
        w = tmp_path / f"rec{i}.wav"
        _make_recording(str(w))
        lines.append(f"rec{i} {w}")
    scp.write_text("\n".join(lines) + "\n")

    full = tmp_path / "grid_full.tsv"
    score_grid(str(scp), str(full), target_sr=48000, window_s=10.0)
    ids_full = {l.split("\t")[1] for l in full.read_text().splitlines()[1:]}

    ids_sharded = set()
    for sid in range(2):
        s = tmp_path / f"grid_{sid}.tsv"
        score_grid(str(scp), str(s), target_sr=48000, window_s=10.0,
                   num_shards=2, shard_id=sid)
        ids_sharded |= {l.split("\t")[1] for l in s.read_text().splitlines()[1:]}

    assert ids_sharded == ids_full == {"rec0", "rec1", "rec2", "rec3"}
