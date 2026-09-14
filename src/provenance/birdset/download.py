#!/usr/bin/env python3
"""Download one BirdSet soundscape split from HuggingFace into the dump tree.

BirdSet (``DBD-research-group/BirdSet``) ships, per region config (HSN, POW, PER,
NES, NBP, SNE, SSW, UHH), a strongly-labelled soundscape **``test``** split: full
soundscape recordings with per-event ``start_time``/``end_time`` and an ``ebird_code``.
That is exactly the *master tape + event times* structure we need, so this script
materializes it into a small, schema-stable **dump** that ``build_eval_set.py`` then
cuts clips from offline:

    download/birdset/<config>/
      soundscapes/<master_id>.<ext>    # unique full soundscape recordings (masters)
      annotations.tsv                  # master_id  start_s  end_s  ebird_code  (one row/event)
      meta.tsv                         # config, split, n_masters, n_events

Design notes
------------
* BirdSet is a **script-based** dataset (ships ``BirdSet.py``), so it needs
  ``datasets<=3.6.0`` and ``trust_remote_code=True`` -- newer ``datasets`` refuse it
  ("Dataset scripts are no longer supported"). We pin that in ``pyproject.toml``.
* ``ebird_code`` is a ``ClassLabel`` -> the raw field is an **int**; we resolve it back
  to the species string via the dataset's class names before writing ``annotations.tsv``.
* ``load_dataset`` runs ``download_and_prepare``, which downloads & extracts **every split of
  the loaded config**. We therefore default to the **``<region>_scape``** config (``--scape``),
  which has only the soundscape ``test``/``test_5s`` splits -- *not* the multi-GB focal
  ``train`` -- so a region costs a few GB instead of ~14 GB. ``--no-scape`` loads the full
  ``<region>`` config (focal train included). The dump dir is named by the bare region either
  way. The HF cache (``.hf_cache``) is disposable once the dump is built (``soundscapes/`` holds
  independent copies); ``--limit`` only caps materialization, not the download.
* Audio is copied with ``Audio(decode=False)`` -- we never decode/resample here, just
  write each unique soundscape's bytes (or copy its cached file) once; decoding/resampling
  is the builder's job. The HuggingFace cache must live on scratch
  (``--cache-dir`` / ``HF_HOME``), never ``$HOME``.

Standalone: imports ``datasets`` lazily inside :func:`download` so the pure helpers
(and unit tests) import without the dep. Run via ``run_download.sbatch`` (BirdSet is
large). See ``src/provenance/birdset/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil

REPO = "DBD-research-group/BirdSet"


def master_id_from_filepath(filepath: str) -> str:
    """Per-soundscape id = recording **basename** (no extension), sanitized.

    BirdSet's ``filepath`` is an absolute extracted path, so we key on the basename;
    BirdSet recording stems (e.g. ``HSN_038_20150710_082105``) are globally unique, so
    this is a clean, stable id. :func:`download` additionally fails loud on the
    (theoretical) case of two distinct soundscapes sharing a basename.
    """
    base = os.path.splitext(os.path.basename(str(filepath).replace("\\", "/")))[0]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in base)


def hf_config_name(config: str, scape: bool = True) -> str:
    """HF config to load: the soundscape-only ``<region>_scape`` (no focal train) by default."""
    if scape and not config.endswith("_scape"):
        return f"{config}_scape"
    return config


def _default_decode(v):
    """Fallback code resolver (no class map): pass strings, drop None/empty/ints."""
    if v is None or v == "" or isinstance(v, (int, bool)):
        return None if not isinstance(v, str) else (v or None)
    return str(v)


def event_rows_from_example(ex: dict, decode_code=None):
    """Yield ``(start_s, end_s, species)`` events from one BirdSet test example.

    ``decode_code`` maps a raw label value to a species string (used to turn the
    ``ClassLabel`` **int** in ``ebird_code`` into its name; :func:`download` supplies
    one built from the dataset's class names). Tolerant of the two label spellings: a
    single ``ebird_code`` (the ``test`` split) or a list ``ebird_code_multilabel``.
    Rows without a usable time span, a resolvable species, or with a 'nocall' label
    are skipped -- they are not positive events.
    """
    decode = decode_code or _default_decode
    start = ex.get("start_time")
    end = ex.get("end_time")
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return
    if not (end > start):
        return

    raw = ex.get("ebird_code")
    if raw is None or raw == "":
        raw_list = ex.get("ebird_code_multilabel") or []
        if isinstance(raw_list, (str, bytes, int)):
            raw_list = [raw_list]
    else:
        raw_list = [raw]
    for rc in raw_list:
        code = decode(rc)
        if code is None or str(code).lower() == "nocall":
            continue
        yield start, end, str(code)


def _class_label(feature):
    """Return a ClassLabel (with ``int2str``) from a ClassLabel or Sequence(ClassLabel)."""
    if feature is None:
        return None
    if hasattr(feature, "int2str"):
        return feature
    inner = getattr(feature, "feature", None)
    if inner is not None and hasattr(inner, "int2str"):
        return inner
    return None


def make_code_decoder(features):
    """Build a ``decode_code`` that resolves ``ebird_code`` ints to species names."""
    cl = _class_label(features.get("ebird_code")) or _class_label(
        features.get("ebird_code_multilabel")
    )

    def decode(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            if v < 0 or cl is None:
                return None
            try:
                return cl.int2str(v)
            except (ValueError, IndexError):
                return None
        if v is None or v == "":
            return None
        return str(v)

    return decode


def _example_audio_path(ex: dict) -> str | None:
    audio = ex.get("audio")
    if isinstance(audio, dict):
        return audio.get("path")
    return None


def download(config: str, split: str, out_dir: str, cache_dir: str | None = None,
             limit: int | None = None, scape: bool = True) -> dict:
    """Materialize ``config``'s ``split`` into ``out_dir/<config>/`` (see module doc).

    With ``scape=True`` (default) the **``<config>_scape``** HF config is loaded (soundscapes
    only, no focal train), but the dump dir is still named by the bare ``config``.
    Returns a small summary dict ``{n_masters, n_events}``.
    """
    from datasets import Audio, get_dataset_split_names, load_dataset  # lazy

    # BirdSet is script-based: needs trust_remote_code. Fall back if the running
    # datasets version doesn't accept the kwarg (so the helper stays version-robust).
    def _hf(fn, *args, **kw):
        try:
            return fn(*args, trust_remote_code=True, **kw)
        except TypeError:
            return fn(*args, **kw)

    hf_config = hf_config_name(config, scape)
    names = _hf(get_dataset_split_names, REPO, hf_config)
    if split not in names:
        raise SystemExit(
            f"split {split!r} not available for config {hf_config!r}; available: {names}"
        )

    ds = _hf(load_dataset, REPO, hf_config, split=split, cache_dir=cache_dir)
    # decode=False -> ex['audio'] is {'path':..., 'bytes':...}; no decode/resample.
    ds = ds.cast_column("audio", Audio(decode=False))
    decode_code = make_code_decoder(ds.features)   # ClassLabel int -> species name

    base = os.path.join(out_dir, config)
    sounds = os.path.join(base, "soundscapes")
    os.makedirs(sounds, exist_ok=True)

    seen: dict[str, str] = {}   # master_id -> source filepath (for collision detection)
    n_events = 0
    with open(os.path.join(base, "annotations.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["master_id", "start_s", "end_s", "ebird_code"])
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            filepath = ex.get("filepath") or _example_audio_path(ex) or f"rec{i}"
            mid = master_id_from_filepath(filepath)

            if mid not in seen:
                audio = ex.get("audio") if isinstance(ex.get("audio"), dict) else {}
                data_bytes = audio.get("bytes")
                src = audio.get("path")
                ext = os.path.splitext(src or str(filepath))[1] or ".ogg"
                dst = os.path.join(sounds, f"{mid}{ext}")
                if data_bytes:                       # prefer in-memory bytes
                    with open(dst, "wb") as af:
                        af.write(data_bytes)
                elif src and os.path.exists(src):    # else copy the cached file
                    shutil.copyfile(src, dst)
                else:
                    raise SystemExit(f"no audio bytes/path for example {i} ({filepath})")
                seen[mid] = filepath
            elif seen[mid] != filepath:
                raise SystemExit(
                    f"master_id collision {mid!r}: {filepath!r} vs {seen[mid]!r} -- "
                    "basenames not unique for this config; adjust master_id_from_filepath"
                )

            for start, end, code in event_rows_from_example(ex, decode_code):
                w.writerow([mid, f"{start:.4f}", f"{end:.4f}", code])
                n_events += 1

    summary = {"n_masters": len(seen), "n_events": n_events}
    with open(os.path.join(base, "meta.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["config", "split", "n_masters", "n_events"])
        w.writerow([config, split, summary["n_masters"], summary["n_events"]])
    print(f"[birdset.download] {hf_config}:{split} -> dump {config}: "
          f"{summary['n_masters']} masters, {summary['n_events']} events -> {base}/")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="BirdSet region, e.g. HSN, POW, PER")
    ap.add_argument("--split", default="test", help="strongly-labelled split (default: test)")
    ap.add_argument("--out", default="download/birdset", help="dump root")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache dir (keep off a quota-limited $HOME)")
    ap.add_argument("--limit", type=int, default=None, help="cap examples (sanity runs)")
    ap.add_argument("--scape", action=argparse.BooleanOptionalAction, default=True,
                    help="load the <region>_scape config (soundscapes only, no focal train)")
    args = ap.parse_args(argv)
    download(args.config, args.split, args.out, cache_dir=args.cache_dir,
             limit=args.limit, scape=args.scape)


if __name__ == "__main__":
    main()
