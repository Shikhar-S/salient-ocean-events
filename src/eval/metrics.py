"""Event-based metrics: precision / recall / F1 and average precision.

Given per-file predicted and ground-truth events plus a :class:`~src.eval.matching.Matcher`,
this computes:

* **AP (PR-AUC)** -- sweep the per-event score over all thresholds, accumulating
  one-to-one matches, and integrate the precision-recall curve. This is the primary
  threshold-free, cross-method number.
* **max-F1** -- the best F1 reachable along that same sweep, with the precision,
  recall, and score threshold at which it occurs.
* **Fixed operating point** -- P/R/F1 with *every* prediction kept (micro, pooled
  over files) plus **macro-F1**, the mean of per-file F1 at that same point.

Matching is one-to-one and greedy: predictions are considered in descending score
order, and each is matched to the still-unmatched ground-truth event in the same
file with the highest matcher quality. A prediction can only match ground truth
from its own ``file_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.types import Event
from src.eval.matching import Matcher


@dataclass
class EvalResult:
    """Container for one detector's scores under one matcher."""

    matcher: str
    n_pred: int
    n_gt: int
    ap: float                 # average precision / PR-AUC (threshold-free)
    max_f1: float             # best F1 along the score sweep
    precision_at_max_f1: float
    recall_at_max_f1: float
    threshold_at_max_f1: float
    # Fixed operating point: every prediction kept.
    precision: float          # micro, pooled across files
    recall: float
    f1: float
    macro_f1: float           # mean of per-file F1
    tp: int
    fp: int
    fn: int
    # Over-firing guards (label-free coverage of the timeline; 0.0 if no durations
    # are supplied). ``selection_rate`` is the keep-all fraction of total audio time
    # flagged as an event; the budgeted recall and AUC reward recall *per unit of
    # coverage* so a fire-everywhere method can't buy recall by firing more.
    selection_rate: float = 0.0          # union pred coverage / total duration (keep-all)
    recall_at_sel_budget: float = 0.0    # recall when capped to selection == budget
    selection_budget: float = 0.15       # the budget used (fraction of timeline)
    auc_recall_sel: float = 0.0          # area under recall-vs-selection over [0, 1]
    # J-optimal operating point (max recall - selection): the deployable threshold.
    op_threshold: float = 0.0
    op_recall: float = 0.0
    op_selection: float = 0.0

    def summary(self) -> str:
        return (
            f"[{self.matcher}] AP={self.ap:.3f} maxF1={self.max_f1:.3f} "
            f"(P={self.precision_at_max_f1:.3f} R={self.recall_at_max_f1:.3f} "
            f"@score>={self.threshold_at_max_f1:.3g})  "
            f"all: P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"macroF1={self.macro_f1:.3f}  TP={self.tp} FP={self.fp} FN={self.fn}  "
            f"sel={self.selection_rate:.3f} "
            f"R@{self.selection_budget:.0%}sel={self.recall_at_sel_budget:.3f} "
            f"AUC(R-sel)={self.auc_recall_sel:.3f}"
        )


def _greedy_match_file(
    preds: list[Event], gts: list[Event], matcher: Matcher
) -> list[bool]:
    """Greedily match one file's predictions (score desc) to its ground truth.

    Returns a list of TP/FP flags aligned to ``preds`` *sorted by descending
    score* (the returned order matches that sorted order, not the input order).
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    used = [False] * len(gts)
    flags: list[bool] = []
    for i in order:
        p = preds[i]
        best_j, best_q = -1, None
        for j, g in enumerate(gts):
            if used[j]:
                continue
            q = matcher(p, g)
            if q is None:
                continue
            if best_q is None or q > best_q:
                best_q, best_j = q, j
        if best_j >= 0:
            used[best_j] = True
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _selection_metrics(
    scored: list[tuple[float, bool]],
    n_gt: int,
    pred_by_file: dict[str, list[Event]],
    durations: dict[str, float] | None,
    budget: float,
    grid_s: float,
) -> tuple[float, float, float, float, float, float]:
    """Label-free over-firing guards + the J-optimal operating point.

    Returns ``(selection_rate, recall@budget, auc, op_threshold, op_recall,
    op_selection)``. ``selection_rate`` is the keep-all union coverage of all
    predictions over total duration. Sweeping the score threshold gives a monotone
    recall-vs-selection curve; ``recall@budget`` is the recall at ``budget``
    coverage and ``auc`` is the area under it over selection in ``[0, 1]`` (recall
    held at its keep-all max beyond reach). The **J-optimal operating point** is the
    threshold maximizing ``recall - selection`` (ROC-style: selection ~ false-
    activity rate since most of the timeline is non-event) -- the deployable point
    of "highest recall, least coverage". All reward recall *per unit of coverage*,
    so firing more cannot inflate them, and need no ground truth for selection.
    """
    if not durations:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Per-bin maximum prediction score on a uniform time grid; a bin counts as
    # covered at threshold t iff some prediction with score >= t spans it.
    total_bins = 0
    covered: list[np.ndarray] = []
    for f, dur in durations.items():
        nb = max(1, int(round(dur / grid_s)))
        total_bins += nb
        preds = pred_by_file.get(f, [])
        if not preds:
            continue
        grid = np.full(nb, -np.inf)
        for p in preds:
            a = max(0, int(p.onset_s / grid_s))
            b = min(nb, int(np.ceil(p.offset_s / grid_s)))
            if b > a:
                grid[a:b] = np.maximum(grid[a:b], p.score)
        covered.append(grid[np.isfinite(grid)])
    if total_bins == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    cov = np.concatenate(covered) if covered else np.array([])
    selection_rate = len(cov) / total_bins
    if cov.size == 0 or n_gt == 0:
        return selection_rate, 0.0, 0.0, 0.0, 0.0, 0.0

    cov_sorted = np.sort(cov)  # ascending; selection(t) = frac of bins with score >= t

    def selection_at(t: float) -> float:
        idx = int(np.searchsorted(cov_sorted, t, side="left"))
        return (len(cov_sorted) - idx) / total_bins

    # Recall-vs-selection points (selection, recall, threshold), threshold
    # decreasing so both selection and recall rise.
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, float("inf"))]
    cum_tp, i, n = 0, 0, len(scored)
    while i < n:
        s = scored[i][0]
        j = i
        while j < n and scored[j][0] == s:
            if scored[j][1]:
                cum_tp += 1
            j += 1
        pts.append((selection_at(s), cum_tp / n_gt, s))
        i = j

    # recall at the budget (linear interp between bracketing points).
    recall_at_budget = pts[-1][1]
    prev_sel, prev_rec = 0.0, 0.0
    for sel, rec, _thr in pts:
        if sel >= budget:
            span = sel - prev_sel
            recall_at_budget = rec if span <= 0 else prev_rec + (
                (budget - prev_sel) / span
            ) * (rec - prev_rec)
            break
        prev_sel, prev_rec = sel, rec

    # AUC of recall over selection in [0, 1]; hold recall at its max past s_max.
    auc = 0.0
    for k in range(1, len(pts)):
        s0, r0 = pts[k - 1][0], pts[k - 1][1]
        s1, r1 = pts[k][0], pts[k][1]
        auc += (s1 - s0) * (r0 + r1) / 2.0
    s_max, r_max = pts[-1][0], pts[-1][1]
    auc += r_max * (1.0 - s_max)

    # J-optimal operating point: max (recall - selection) over the curve.
    op_sel, op_rec, op_thr = max(pts, key=lambda p: p[1] - p[0])
    return selection_rate, recall_at_budget, auc, op_thr, op_rec, op_sel


def evaluate(
    pred_by_file: dict[str, list[Event]],
    gt_by_file: dict[str, list[Event]],
    matcher: Matcher,
    durations: dict[str, float] | None = None,
    selection_budget: float = 0.15,
    grid_s: float = 0.01,
) -> EvalResult:
    """Evaluate predictions against ground truth under ``matcher``.

    If ``durations`` (``{file_id: seconds}``) is given, also compute the label-free
    over-firing guards (selection rate, recall at the ``selection_budget`` coverage,
    and the recall-vs-selection AUC).
    """
    files = set(pred_by_file) | set(gt_by_file)
    n_gt = sum(len(gt_by_file.get(f, [])) for f in files)
    n_pred = sum(len(pred_by_file.get(f, [])) for f in files)

    # --- Global score-ordered sweep (pooled across files) for AP and max-F1. ---
    # Build (score, tp_flag) by matching greedily in global score order, with the
    # one-to-one constraint enforced per file.
    scored: list[tuple[float, bool]] = []
    used: dict[str, list[bool]] = {
        f: [False] * len(gt_by_file.get(f, [])) for f in files
    }
    # Flatten predictions tagged with their file, sort by score desc (stable).
    flat = [
        (p.score, f, p)
        for f in files
        for p in pred_by_file.get(f, [])
    ]
    flat.sort(key=lambda t: t[0], reverse=True)
    for score, f, p in flat:
        gts = gt_by_file.get(f, [])
        best_j, best_q = -1, None
        for j, g in enumerate(gts):
            if used[f][j]:
                continue
            q = matcher(p, g)
            if q is None:
                continue
            if best_q is None or q > best_q:
                best_q, best_j = q, j
        if best_j >= 0:
            used[f][best_j] = True
            scored.append((score, True))
        else:
            scored.append((score, False))

    ap, max_f1, p_at, r_at, thr_at = _sweep_metrics(scored, n_gt)

    # --- Fixed operating point: keep every prediction (micro, pooled). ---
    tp = sum(1 for _, is_tp in scored if is_tp)
    fp = n_pred - tp
    fn = n_gt - tp
    precision, recall, f1 = _f1(tp, fp, fn)

    # --- Macro-F1: mean per-file F1 at the keep-everything operating point. ---
    per_file_f1: list[float] = []
    for f in files:
        preds = pred_by_file.get(f, [])
        gts = gt_by_file.get(f, [])
        if not preds and not gts:
            per_file_f1.append(1.0)  # nothing to find, nothing predicted
            continue
        flags = _greedy_match_file(preds, gts, matcher)
        ftp = sum(flags)
        ffp = len(preds) - ftp
        ffn = len(gts) - ftp
        per_file_f1.append(_f1(ftp, ffp, ffn)[2])
    macro_f1 = sum(per_file_f1) / len(per_file_f1) if per_file_f1 else 0.0

    (
        selection_rate, recall_at_sel_budget, auc_recall_sel,
        op_threshold, op_recall, op_selection,
    ) = _selection_metrics(
        scored, n_gt, pred_by_file, durations, selection_budget, grid_s
    )

    return EvalResult(
        matcher=getattr(matcher, "name", type(matcher).__name__),
        n_pred=n_pred,
        n_gt=n_gt,
        ap=ap,
        max_f1=max_f1,
        precision_at_max_f1=p_at,
        recall_at_max_f1=r_at,
        threshold_at_max_f1=thr_at,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro_f1,
        tp=tp,
        fp=fp,
        fn=fn,
        selection_rate=selection_rate,
        recall_at_sel_budget=recall_at_sel_budget,
        selection_budget=selection_budget,
        auc_recall_sel=auc_recall_sel,
        op_threshold=op_threshold,
        op_recall=op_recall,
        op_selection=op_selection,
    )


def _sweep_metrics(
    scored: list[tuple[float, bool]], n_gt: int
) -> tuple[float, float, float, float, float]:
    """From score-ordered (score, tp_flag) pairs compute AP and the max-F1 point.

    Returns ``(ap, max_f1, precision_at_max_f1, recall_at_max_f1, threshold)``.
    AP integrates the PR curve as ``sum_n (R_n - R_{n-1}) * P_n`` (sklearn-style).
    """
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if not scored:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # scored is already in descending score order from the caller.
    ap = 0.0
    prev_recall = 0.0
    cum_tp = 0
    cum_fp = 0
    best_f1, best_p, best_r, best_thr = 0.0, 0.0, 0.0, scored[0][0]

    n = len(scored)
    i = 0
    while i < n:
        # Process all predictions sharing this score together (a single operating
        # point): a threshold cannot separate equal scores.
        score = scored[i][0]
        j = i
        while j < n and scored[j][0] == score:
            if scored[j][1]:
                cum_tp += 1
            else:
                cum_fp += 1
            j += 1
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / n_gt
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, precision, recall, score
        i = j

    return ap, best_f1, best_p, best_r, best_thr
