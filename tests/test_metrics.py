"""Event-based metrics on synthetic events with hand-computed expected values."""

import math

import pytest

from src.core.types import Event
from src.eval.matching import CollarMatcher, IoUMatcher
from src.eval.metrics import evaluate


def _gt():
    return {"f": [Event(0, 1), Event(2, 3), Event(5, 6)]}


def test_collar_basic_counts_and_ap():
    preds = {
        "f": [
            Event(0.05, 1.05, score=0.9),  # matches GT[0]
            Event(2.00, 3.00, score=0.8),  # matches GT[1]
            Event(10.0, 11.0, score=0.7),  # no GT  -> FP
            # GT[2] [5,6] left unmatched -> FN
        ]
    }
    r = evaluate(preds, _gt(), CollarMatcher())
    assert (r.n_gt, r.n_pred) == (3, 3)
    assert (r.tp, r.fp, r.fn) == (2, 1, 1)
    assert r.precision == pytest.approx(2 / 3)
    assert r.recall == pytest.approx(2 / 3)
    assert r.f1 == pytest.approx(2 / 3)
    assert r.ap == pytest.approx(2 / 3)
    assert r.max_f1 == pytest.approx(0.8)
    assert r.threshold_at_max_f1 == pytest.approx(0.8)


def test_one_to_one_matching_no_double_count():
    # Two predictions both compatible with the single GT[0]; only one may match.
    preds = {"f": [Event(0.0, 1.0, score=0.9), Event(0.1, 1.0, score=0.5)]}
    gt = {"f": [Event(0, 1)]}
    r = evaluate(preds, gt, CollarMatcher())
    assert (r.tp, r.fp, r.fn) == (1, 1, 0)


def test_file_isolation():
    # A prediction in file 'a' must not match ground truth in file 'b'.
    preds = {"a": [Event(0, 1, score=0.9)]}
    gt = {"b": [Event(0, 1)]}
    r = evaluate(preds, gt, CollarMatcher())
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)


def test_iou_threshold_rejects_low_overlap():
    gt = {"f": [Event(0, 1)]}
    # 50% overlap -> IoU = 0.5/1.5 = 0.333 < 0.5 -> no match.
    preds = {"f": [Event(0.5, 1.5, score=1.0)]}
    r = evaluate(preds, gt, IoUMatcher(threshold=0.5))
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)
    # Tight overlap matches.
    preds2 = {"f": [Event(0.0, 1.0, score=1.0)]}
    r2 = evaluate(preds2, gt, IoUMatcher(threshold=0.5))
    assert (r2.tp, r2.fp, r2.fn) == (1, 0, 0)


def test_perfect_predictions():
    gt = _gt()
    preds = {"f": [Event(e.onset_s, e.offset_s, score=1.0) for e in gt["f"]]}
    r = evaluate(preds, gt, CollarMatcher())
    assert r.ap == pytest.approx(1.0)
    assert r.f1 == pytest.approx(1.0)
    assert r.macro_f1 == pytest.approx(1.0)


def test_no_ground_truth():
    preds = {"f": [Event(0, 1, score=0.9)]}
    r = evaluate(preds, {"f": []}, CollarMatcher())
    assert r.n_gt == 0
    assert r.ap == 0.0
    assert math.isclose(r.fp, 1)


def test_selection_rate_and_no_durations():
    gt = {"f": [Event(2, 3)]}
    preds = {"f": [Event(0, 1, score=0.9), Event(2, 3, score=0.8)]}  # 2s of 10s
    # Without durations the guards are inert.
    r0 = evaluate(preds, gt, IoUMatcher())
    assert r0.selection_rate == 0.0 and r0.auc_recall_sel == 0.0
    # With durations, selection_rate = covered/total = 2/10.
    r = evaluate(preds, gt, IoUMatcher(), durations={"f": 10.0})
    assert r.selection_rate == pytest.approx(0.2, abs=1e-3)


def test_selection_catches_over_firing():
    # One real event at [5,6] in a 10s clip. A *precise* detector fires only there;
    # an *over-firing* detector also blankets the clip with junk and "recovers" the
    # event too. Both get recall=1, but selection rate / budgeted recall separate them.
    gt = {"f": [Event(5, 6)]}
    durations = {"f": 10.0}

    precise = {"f": [Event(5.0, 6.0, score=1.0)]}
    # Undiscriminative over-firer: junk scored the same as the real event (the
    # realistic case that also tanks AP), so the threshold sweep can't filter it.
    overfire = {"f": [Event(5.0, 6.0, score=1.0)]
                + [Event(float(t), t + 0.9, score=1.0) for t in range(0, 10) if t != 5]}

    rp = evaluate(precise, gt, IoUMatcher(), durations=durations)
    ro = evaluate(overfire, gt, IoUMatcher(), durations=durations)

    assert rp.recall == pytest.approx(1.0) and ro.recall == pytest.approx(1.0)
    # Over-firer covers most of the timeline; precise covers ~10%.
    assert rp.selection_rate < 0.2 < ro.selection_rate
    # The guards reward recall-per-coverage, so the precise detector wins on both
    # the budgeted recall and the AUC even though raw recall ties.
    assert rp.recall_at_sel_budget > ro.recall_at_sel_budget
    assert rp.auc_recall_sel > ro.auc_recall_sel


def test_j_optimal_operating_point():
    gt = {"f": [Event(5, 6)]}
    durations = {"f": 10.0}
    precise = {"f": [Event(5.0, 6.0, score=1.0)]}
    overfire = {"f": [Event(5.0, 6.0, score=1.0)]
                + [Event(float(t), t + 0.9, score=1.0) for t in range(0, 10) if t != 5]}
    rp = evaluate(precise, gt, IoUMatcher(), durations=durations)
    ro = evaluate(overfire, gt, IoUMatcher(), durations=durations)
    # Precise detector: catches the event at low coverage -> J ~ 0.9 at a real threshold.
    assert rp.op_recall == pytest.approx(1.0)
    assert rp.op_selection < 0.2
    assert rp.op_threshold <= 1.0
    # J = recall - selection is higher for the precise detector.
    assert (rp.op_recall - rp.op_selection) > (ro.op_recall - ro.op_selection)
