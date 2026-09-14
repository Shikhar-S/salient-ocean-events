"""Tests for the BirdSet eval-set builder, driven by a tiny synthetic dump.

The builder is decoupled from HuggingFace via the on-disk dump contract
(`soundscapes/*` + `annotations.tsv`), so we fabricate a dump, build, and assert the
output matches the selection-table contract the benchmark consumes.
"""

import csv
import os

import numpy as np
import pytest
import soundfile as sf

from src.provenance.birdset import build_eval_set as B


def test_merge_intervals_unions_overlaps():
    assert B.merge_intervals([(3.0, 3.6), (0.0, 1.0), (3.5, 4.0)]) == [(0.0, 1.0), (3.0, 4.0)]
    assert B.merge_intervals([]) == []


def test_events_in_window_clips_and_merges():
    events = [(1.0, 1.4, "A"), (3.0, 3.6, "B"), (3.5, 4.0, "A")]
    # window [0.5, 4.5): event1 -> [0.5,0.9]; events 2&3 overlap -> merged [2.5,3.5]
    spans = B.events_in_window(events, 0.5, 4.5)
    flat = [v for span in spans for v in span]
    assert flat == pytest.approx([0.5, 0.9, 2.5, 3.5])


def _write_master(path, sr, dur_s, events, seed=0):
    """A silent master with a short noise burst at each event (so audio is non-trivial)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur_s * sr), dtype=np.float32)
    for s, e, _c in events:
        a, b = int(s * sr), int(e * sr)
        x[a:b] = 0.3 * rng.standard_normal(b - a).astype(np.float32)
    sf.write(path, x, sr)


def _make_dump(root):
    base = os.path.join(root, "TEST")
    os.makedirs(os.path.join(base, "soundscapes"), exist_ok=True)
    m1 = [(1.0, 1.4, "A"), (3.0, 3.6, "B"), (3.5, 4.0, "A")]  # last two overlap -> merge
    m2 = [(2.0, 2.5, "C")]
    _write_master(os.path.join(base, "soundscapes", "m1.wav"), 32000, 12.0, m1, 1)
    _write_master(os.path.join(base, "soundscapes", "m2.wav"), 16000, 12.0, m2, 2)  # resample path
    with open(os.path.join(base, "annotations.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["master_id", "start_s", "end_s", "ebird_code"])
        for mid, evs in (("m1", m1), ("m2", m2)):
            for s, e, c in evs:
                w.writerow([mid, f"{s:.4f}", f"{e:.4f}", c])
    return root


def _build(tmp_path, **over):
    dump = _make_dump(str(tmp_path / "dump"))
    out = str(tmp_path / "data" / "birdset")
    argv = ["--dump", dump, "--config", "TEST", "--out", out,
            "--clip-s", "2.0", "--margin-s", "0.2", "--target-sr", "32000",
            "--max-positives", "10", "--max-per-species", "10", "--seed", "0"]
    for k, v in over.items():
        argv += [f"--{k}", str(v)]
    B.main(argv)
    return out


def _read_meta(out):
    with open(os.path.join(out, "metadata.csv")) as fh:
        return list(csv.DictReader(fh))


def _read_gt(out):
    with open(os.path.join(out, "gt.tsv")) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_build_writes_contract_artifacts(tmp_path):
    out = _build(tmp_path)
    assert os.path.exists(os.path.join(out, "gt.tsv"))
    assert os.path.exists(os.path.join(out, "metadata.csv"))
    assert os.path.exists(os.path.join(out, "README.md"))
    meta = _read_meta(out)
    # 4 annotated events across 2 masters -> 4 positives.
    pos = [m for m in meta if m["label"] == "event"]
    neg = [m for m in meta if m["label"] == "none"]
    assert len(pos) == 4
    assert len(neg) >= 1
    # every clip file actually exists and is exactly clip-s * sr samples, mono.
    for m in meta:
        p = os.path.join(out, m["file_name"])
        info = sf.info(p)
        assert info.samplerate == 32000 and info.channels == 1
        assert info.frames == int(2.0 * 32000)


def test_gt_only_for_positives_and_within_clip(tmp_path):
    out = _build(tmp_path)
    meta = _read_meta(out)
    pos_ids = {os.path.splitext(os.path.basename(m["file_name"]))[0]
               for m in meta if m["label"] == "event"}
    neg_ids = {os.path.splitext(os.path.basename(m["file_name"]))[0]
               for m in meta if m["label"] == "none"}
    gt = _read_gt(out)
    assert gt, "expected some GT rows"
    by_id = {}
    for r in gt:
        fid = r["file_id"]
        b, e = float(r["Begin Time (s)"]), float(r["End Time (s)"])
        assert fid in pos_ids and fid not in neg_ids       # GT only on positives
        assert 0.0 <= b < e <= 2.0                          # within the clip, well-ordered
        by_id.setdefault(fid, []).append((b, e))
    # merged GT spans within a clip never overlap.
    for spans in by_id.values():
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0


def test_species_cap_limits_positives(tmp_path):
    # cap 1 per species over species {A,B,C} -> at most 3 positives.
    out = _build(tmp_path, **{"max-per-species": 1})
    pos = [m for m in _read_meta(out) if m["label"] == "event"]
    assert len(pos) <= 3
    assert len({m["species"] for m in pos}) == len(pos)    # distinct species
