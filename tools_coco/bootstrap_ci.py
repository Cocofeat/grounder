#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for all evaluation metrics.

Supports both single-modal and multi-modal evaluations.
Resamples at the patient level (or patient-grouped slices for grounding).

Usage:
    python tools_coco/bootstrap_ci.py --mode single --n-iterations 1000
    python tools_coco/bootstrap_ci.py --mode multi  --n-iterations 1000
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = Path("/home/baiyang/nltk_data")
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))

# Suppress verbose pycocoevalcap output
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import configs and helpers from existing summarize scripts
# ---------------------------------------------------------------------------
sys.path.insert(0, str(BASE_DIR))

from tools_coco.summarize_all_results import (
    DATASETS as MULTI_DATASETS,
    MERGED_DATASETS as MULTI_MERGED,
    load_jsonl,
    majority_vote,
    compute_f1,
    _extract_patient_id_qa,
    extract_patient_id_report,
)
from tools_coco.summarize_all_results_single import (
    DATASETS as SINGLE_DATASETS,
    MERGED_DATASETS as SINGLE_MERGED,
    _extract_patient_id_single,
    parse_bounding_box_single,
    calculate_iou,
)


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------

def bootstrap_ci(unit_ids, get_metrics_fn, n_iter=1000, ci_level=0.95, seed=42,
                 label=""):
    """Compute bootstrap confidence intervals.

    Args:
        unit_ids: list of resampling unit IDs (patients or slices)
        get_metrics_fn: callable(sampled_ids) -> dict[str, float]
        n_iter: number of bootstrap iterations
        ci_level: confidence level (default 0.95)
        seed: random seed
        label: label for tqdm progress bar

    Returns:
        dict of {metric_name: {point, ci_lower, ci_upper}}
    """
    rng = np.random.RandomState(seed)
    n = len(unit_ids)
    if n == 0:
        return {}
    unit_arr = np.array(unit_ids)

    # Point estimate on full data
    point = get_metrics_fn(unit_ids)

    # Bootstrap
    boot_results = defaultdict(list)
    for _ in tqdm(range(n_iter), desc=f"    {label}", leave=False, ncols=80):
        sampled = rng.choice(unit_arr, size=n, replace=True).tolist()
        metrics = get_metrics_fn(sampled)
        for k, v in metrics.items():
            boot_results[k].append(v)

    alpha = (1 - ci_level) / 2
    results = {}
    for k in point:
        arr = np.array(boot_results[k])
        results[k] = {
            "point": point[k],
            "ci_lower": float(np.percentile(arr, alpha * 100)),
            "ci_upper": float(np.percentile(arr, (1 - alpha) * 100)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return results


# ---------------------------------------------------------------------------
# QA data preparation
# ---------------------------------------------------------------------------

def prepare_qa_multi(gt_path, pred_path, patient_filter=None):
    """Prepare multi-modal QA data for bootstrap.

    Returns (patient_ids, get_metrics_fn).
    patient_data[pid][qtype] = (voted_pred, gt_label)
    """
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    pred_by_qid = {p["question_id"]: p for p in preds}

    raw = defaultdict(lambda: defaultdict(lambda: {"preds": [], "gt": None}))
    for gt in gts:
        qid = gt["question_id"]
        pid = _extract_patient_id_qa(gt)
        if patient_filter and pid not in patient_filter:
            continue
        if qid not in pred_by_qid:
            continue
        qtype = gt["Question_type"]
        raw[pid][qtype]["preds"].append(pred_by_qid[qid]["text"])
        if raw[pid][qtype]["gt"] is None:
            raw[pid][qtype]["gt"] = gt["label"]

    patient_data = {}
    for pid, qtypes in raw.items():
        patient_data[pid] = {}
        for qtype, d in qtypes.items():
            patient_data[pid][qtype] = (majority_vote(d["preds"]), d["gt"])

    patient_ids = sorted(patient_data.keys())

    def get_metrics(sampled_pids):
        all_p, all_g = [], []
        by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
        for pid in sampled_pids:
            if pid not in patient_data:
                continue
            for qtype, (vpred, vgt) in patient_data[pid].items():
                by_qtype[qtype]["pred"].append(vpred)
                by_qtype[qtype]["gt"].append(vgt)
                all_p.append(vpred)
                all_g.append(vgt)
        metrics = {}
        for qtype, d in by_qtype.items():
            if d["gt"]:
                metrics[f"{qtype}_acc"] = sum(
                    1 for a, b in zip(d["pred"], d["gt"]) if a == b
                ) / len(d["gt"])
                metrics[f"{qtype}_f1"] = compute_f1(d["pred"], d["gt"])
        if all_g:
            metrics["Overall_acc"] = sum(
                1 for a, b in zip(all_p, all_g) if a == b
            ) / len(all_g)
            metrics["Overall_f1"] = compute_f1(all_p, all_g)
        return metrics

    return patient_ids, get_metrics


def prepare_qa_single(gt_path, pred_path, patient_filter=None):
    """Prepare single-modal QA data for bootstrap.

    Uses positional matching (zip) like the original eval script.
    patient_data[pid][(modality, qtype)] = (first_pred, gt_label)

    Returns (patient_ids, get_metrics_fn) where metrics are keyed as
    {mod}_{qtype}_acc, {mod}_Overall_acc, ALL_Overall_acc, etc.
    """
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    # Group by (modality, patient_id, qtype) using positional matching
    raw = defaultdict(lambda: {"pred": [], "gt": []})
    for pred, gt in zip(preds, gts):
        pid = _extract_patient_id_single(gt)
        if patient_filter and pid not in patient_filter:
            continue
        mod = gt.get("modality", "unknown")
        qtype = gt["Question_type"]
        raw[(mod, pid, qtype)]["pred"].append(pred["text"])
        raw[(mod, pid, qtype)]["gt"].append(gt["label"])

    # First-slice selection per (mod, pid, qtype)
    patient_data = defaultdict(dict)
    for (mod, pid, qtype), data in raw.items():
        patient_data[pid][(mod, qtype)] = (data["pred"][0], data["gt"][0])

    patient_ids = sorted(patient_data.keys())

    def get_metrics(sampled_pids):
        by_mod_qtype = defaultdict(lambda: {"pred": [], "gt": []})
        for pid in sampled_pids:
            if pid not in patient_data:
                continue
            for (mod, qtype), (vpred, vgt) in patient_data[pid].items():
                by_mod_qtype[(mod, qtype)]["pred"].append(vpred)
                by_mod_qtype[(mod, qtype)]["gt"].append(vgt)

        metrics = {}
        # Per modality
        modalities = sorted(set(m for (m, _) in by_mod_qtype.keys()))
        for mod in modalities:
            mod_p, mod_g = [], []
            for qtype in sorted(set(q for (m, q) in by_mod_qtype.keys() if m == mod)):
                d = by_mod_qtype[(mod, qtype)]
                if d["gt"]:
                    metrics[f"{mod}_{qtype}_acc"] = sum(
                        1 for a, b in zip(d["pred"], d["gt"]) if a == b
                    ) / len(d["gt"])
                    metrics[f"{mod}_{qtype}_f1"] = compute_f1(d["pred"], d["gt"])
                    mod_p.extend(d["pred"])
                    mod_g.extend(d["gt"])
            if mod_g:
                metrics[f"{mod}_Overall_acc"] = sum(
                    1 for a, b in zip(mod_p, mod_g) if a == b
                ) / len(mod_g)
                metrics[f"{mod}_Overall_f1"] = compute_f1(mod_p, mod_g)

        # ALL combined
        all_p, all_g = [], []
        all_by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
        for (mod, qtype), d in by_mod_qtype.items():
            all_by_qtype[qtype]["pred"].extend(d["pred"])
            all_by_qtype[qtype]["gt"].extend(d["gt"])
            all_p.extend(d["pred"])
            all_g.extend(d["gt"])
        for qtype, d in all_by_qtype.items():
            if d["gt"]:
                metrics[f"ALL_{qtype}_acc"] = sum(
                    1 for a, b in zip(d["pred"], d["gt"]) if a == b
                ) / len(d["gt"])
                metrics[f"ALL_{qtype}_f1"] = compute_f1(d["pred"], d["gt"])
        if all_g:
            metrics["ALL_Overall_acc"] = sum(
                1 for a, b in zip(all_p, all_g) if a == b
            ) / len(all_g)
            metrics["ALL_Overall_f1"] = compute_f1(all_p, all_g)
        return metrics

    return patient_ids, get_metrics


# ---------------------------------------------------------------------------
# Report data preparation
# ---------------------------------------------------------------------------

def _load_report_evaluators():
    """Load pycocoevalcap evaluators once."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    return Bleu(4), Cider(), Rouge()


class _SuppressStdout:
    """Context manager to suppress stdout (e.g. verbose pycocoevalcap output)."""
    def __enter__(self):
        self._fd = os.dup(1)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        return self
    def __exit__(self, *args):
        os.dup2(self._fd, 1)
        os.close(self._fd)
        os.close(self._devnull)


def _compute_report_scores(gt_dict, pred_dict, bleu_eval, cider_eval, rouge_eval):
    """Compute report metrics using pre-instantiated evaluators."""
    if not pred_dict:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0,
                "CIDEr": 0.0, "ROUGE_L": 0.0, "METEOR": 0.0}
    results = {}
    with _SuppressStdout():
        try:
            score, _ = bleu_eval.compute_score(gt_dict, pred_dict)
            results["BLEU-1"] = score[0]
            results["BLEU-2"] = score[1]
            results["BLEU-3"] = score[2]
            results["BLEU-4"] = score[3]
        except Exception:
            results.update({"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0})
        try:
            score, _ = cider_eval.compute_score(gt_dict, pred_dict)
            results["CIDEr"] = score
        except Exception:
            results["CIDEr"] = 0.0
        try:
            score, _ = rouge_eval.compute_score(gt_dict, pred_dict)
            results["ROUGE_L"] = score
        except Exception:
            results["ROUGE_L"] = 0.0
        try:
            from nltk.translate.meteor_score import meteor_score as _nltk_meteor
            meteor_sum = 0.0
            for key in gt_dict:
                ref_tok = gt_dict[key][0].split()
                hyp_tok = pred_dict[key][0].split()
                meteor_sum += _nltk_meteor([ref_tok], hyp_tok)
            results["METEOR"] = meteor_sum / len(gt_dict)
        except Exception:
            results["METEOR"] = 0.0
    return results


_REPORT_METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr", "ROUGE_L", "METEOR"]


def _precompute_report_per_sample(gt_dict, pred_dict, bleu_eval, cider_eval, rouge_eval):
    """Run evaluators ONCE, return per-sample scores {key: {metric: score}}.

    This avoids re-running expensive evaluators on every bootstrap iteration.
    """
    keys = list(gt_dict.keys())
    per_sample = {k: {} for k in keys}
    if not keys:
        return per_sample

    with _SuppressStdout():
        # BLEU — compute_score returns (corpus_scores, per_sample_scores)
        try:
            _, bleu_per = bleu_eval.compute_score(gt_dict, pred_dict)
            for i, n in enumerate([1, 2, 3, 4]):
                for key, s in zip(keys, bleu_per[i]):
                    per_sample[key][f"BLEU-{n}"] = float(s)
        except Exception:
            for key in keys:
                for n in range(1, 5):
                    per_sample[key][f"BLEU-{n}"] = 0.0

        # CIDEr
        try:
            _, cider_per = cider_eval.compute_score(gt_dict, pred_dict)
            for key, s in zip(keys, cider_per):
                per_sample[key]["CIDEr"] = float(s)
        except Exception:
            for key in keys:
                per_sample[key]["CIDEr"] = 0.0

        # ROUGE_L
        try:
            _, rouge_per = rouge_eval.compute_score(gt_dict, pred_dict)
            for key, s in zip(keys, rouge_per):
                per_sample[key]["ROUGE_L"] = float(s)
        except Exception:
            for key in keys:
                per_sample[key]["ROUGE_L"] = 0.0

        # METEOR (nltk — pure Python, no Java)
        try:
            from nltk.translate.meteor_score import meteor_score as _nltk_meteor
            for key in keys:
                ref_tok = gt_dict[key][0].split()
                hyp_tok = pred_dict[key][0].split()
                per_sample[key]["METEOR"] = float(_nltk_meteor([ref_tok], hyp_tok))
        except Exception:
            for key in keys:
                per_sample[key]["METEOR"] = 0.0

    return per_sample


def _build_score_matrix(patient_ids, per_sample_scores, metric_names):
    """Build numpy score matrix from per-sample scores for fast bootstrap.

    Returns (score_matrix, pid_to_idx) where score_matrix[i, j] is
    patient_ids[i]'s score for metric_names[j].
    """
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    score_matrix = np.zeros((len(patient_ids), len(metric_names)))
    for i, pid in enumerate(patient_ids):
        for j, m in enumerate(metric_names):
            score_matrix[i, j] = per_sample_scores.get(pid, {}).get(m, 0.0)
    return score_matrix, pid_to_idx


def _make_fast_get_metrics(score_matrix, pid_to_idx, metric_names):
    """Create a fast get_metrics closure using numpy fancy indexing."""
    def get_metrics(sampled_pids):
        indices = [pid_to_idx[pid] for pid in sampled_pids if pid in pid_to_idx]
        if not indices:
            return {m: 0.0 for m in metric_names}
        sampled = score_matrix[indices]
        means = sampled.mean(axis=0)
        return {metric_names[j]: float(means[j]) for j in range(len(metric_names))}
    return get_metrics


def prepare_report_multi(gt_path, pred_path, evaluators, patient_filter=None):
    """Prepare multi-modal report data for bootstrap.

    Pre-computes per-patient scores once for fast bootstrap resampling.
    Returns (patient_ids, get_metrics_fn).
    """
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    gt_mapping = {gt["question_id"]: gt for gt in gts}
    data_by_patient = defaultdict(lambda: {"preds": [], "gts": []})

    for pred in preds:
        qid = pred["question_id"]
        if qid not in gt_mapping:
            continue
        pid = extract_patient_id_report(qid)
        if patient_filter and str(pid) not in patient_filter:
            continue
        data_by_patient[pid]["preds"].append(pred["text"])
        data_by_patient[pid]["gts"].append(gt_mapping[qid]["label"])

    # First-slice per patient
    patient_report = {}
    for pid, data in data_by_patient.items():
        patient_report[pid] = (data["gts"][0], data["preds"][0])

    patient_ids = sorted(patient_report.keys())
    bleu_eval, cider_eval, rouge_eval = evaluators

    # Pre-compute per-patient scores (run evaluators ONCE)
    gt_dict = {pid: [gt] for pid, (gt, _) in patient_report.items()}
    pred_dict = {pid: [pred] for pid, (_, pred) in patient_report.items()}
    per_sample = _precompute_report_per_sample(
        gt_dict, pred_dict, bleu_eval, cider_eval, rouge_eval)

    score_matrix, pid_to_idx = _build_score_matrix(
        patient_ids, per_sample, _REPORT_METRICS)
    get_metrics = _make_fast_get_metrics(score_matrix, pid_to_idx, _REPORT_METRICS)

    return patient_ids, get_metrics


def prepare_report_single(gt_path, pred_path, evaluators, patient_filter=None):
    """Prepare single-modal report data for bootstrap.

    Pre-computes per-patient scores once for fast bootstrap resampling.
    Returns (patient_ids, get_metrics_fn) with per-modality and ALL metrics.
    """
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    gt_mapping = {gt["question_id"]: gt for gt in gts}
    data_by_mod_patient = defaultdict(lambda: {"preds": [], "gts": []})

    for pred in preds:
        qid = pred["question_id"]
        if qid not in gt_mapping:
            continue
        gt = gt_mapping[qid]
        pid = _extract_patient_id_single(gt)
        if patient_filter and pid not in patient_filter:
            continue
        mod = gt.get("modality", "unknown")
        data_by_mod_patient[(mod, pid)]["preds"].append(pred["text"])
        data_by_mod_patient[(mod, pid)]["gts"].append(gt["label"])

    # First-slice per (mod, patient)
    # patient_report_by_mod[pid][mod] = (gt_text, pred_text)
    patient_report_by_mod = defaultdict(dict)
    for (mod, pid), data in data_by_mod_patient.items():
        patient_report_by_mod[pid][mod] = (data["gts"][0], data["preds"][0])

    patient_ids = sorted(patient_report_by_mod.keys())
    bleu_eval, cider_eval, rouge_eval = evaluators

    # Pre-compute per-patient scores for each modality and ALL (run evaluators ONCE per group)
    # Collect per-patient combined scores: {pid: {"{mod}_BLEU-1": x, ..., "ALL_BLEU-1": y, ...}}
    combined_scores = {pid: {} for pid in patient_ids}

    # Per modality
    modalities = sorted(set(m for pid in patient_ids
                            for m in patient_report_by_mod[pid].keys()))
    for mod in modalities:
        gt_dict, pred_dict = {}, {}
        for pid in patient_ids:
            if mod in patient_report_by_mod[pid]:
                gt_t, pr_t = patient_report_by_mod[pid][mod]
                gt_dict[pid] = [gt_t]
                pred_dict[pid] = [pr_t]
        if gt_dict:
            per_sample = _precompute_report_per_sample(
                gt_dict, pred_dict, bleu_eval, cider_eval, rouge_eval)
            for pid, scores in per_sample.items():
                for m, v in scores.items():
                    combined_scores[pid][f"{mod}_{m}"] = v

    # ALL: use first modality entry per patient
    all_gt_dict, all_pred_dict = {}, {}
    for pid in patient_ids:
        first_mod = next(iter(patient_report_by_mod[pid]))
        gt_t, pr_t = patient_report_by_mod[pid][first_mod]
        all_gt_dict[pid] = [gt_t]
        all_pred_dict[pid] = [pr_t]
    if all_gt_dict:
        per_sample = _precompute_report_per_sample(
            all_gt_dict, all_pred_dict, bleu_eval, cider_eval, rouge_eval)
        for pid, scores in per_sample.items():
            for m, v in scores.items():
                combined_scores[pid][f"ALL_{m}"] = v

    # Build combined metric names and score matrix
    all_metric_names = sorted(set(
        k for pid in patient_ids for k in combined_scores[pid].keys()))
    score_matrix, pid_to_idx = _build_score_matrix(
        patient_ids, combined_scores, all_metric_names)
    get_metrics = _make_fast_get_metrics(score_matrix, pid_to_idx, all_metric_names)

    return patient_ids, get_metrics


# ---------------------------------------------------------------------------
# Grounding data preparation (single-modal only)
# ---------------------------------------------------------------------------

def prepare_grounding_single(gt_path, pred_path, patient_filter=None):
    """Prepare single-modal grounding data for patient-level bootstrap.

    Returns (patient_ids, get_metrics_fn).
    """
    preds_raw = load_jsonl(str(pred_path))
    pred_map = {p["question_id"]: p["text"] for p in preds_raw}

    with open(str(gt_path), "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Build patient_slices[pid] = [(iou, modality), ...]
    patient_slices = defaultdict(list)
    for gt_entry in gt_data:
        qid = gt_entry["question_id"]
        if qid not in pred_map:
            continue
        pid = _extract_patient_id_single(gt_entry)
        if patient_filter and pid not in patient_filter:
            continue
        pred_box = parse_bounding_box_single(pred_map[qid])
        gt_box = parse_bounding_box_single(gt_entry["label"])
        if pred_box is None or gt_box is None:
            continue
        iou = calculate_iou(pred_box, gt_box)
        if iou is None:
            continue
        mod = gt_entry.get("modality", "unknown")
        patient_slices[pid].append((iou, mod))

    patient_ids = sorted(patient_slices.keys())
    iou_thresholds = [0.1, 0.3, 0.5]

    def get_metrics(sampled_pids):
        all_ious = []
        by_mod = defaultdict(list)
        for pid in sampled_pids:
            if pid not in patient_slices:
                continue
            for iou, mod in patient_slices[pid]:
                all_ious.append(iou)
                by_mod[mod].append(iou)

        metrics = {}
        # Per modality
        for mod, ious in by_mod.items():
            arr = np.array(ious)
            metrics[f"{mod}_mIoU"] = float(np.mean(arr))
            for t in iou_thresholds:
                metrics[f"{mod}_IoU@{t}"] = float(np.mean(arr >= t))
        # ALL
        if all_ious:
            arr = np.array(all_ious)
            metrics["ALL_mIoU"] = float(np.mean(arr))
            for t in iou_thresholds:
                metrics[f"ALL_IoU@{t}"] = float(np.mean(arr >= t))
        return metrics

    return patient_ids, get_metrics


# ---------------------------------------------------------------------------
# Merged dataset helpers
# ---------------------------------------------------------------------------

def prepare_merged_qa_multi(sources, datasets, task_key, patient_filter=None):
    """Merge QA data from multiple source datasets for multi-modal bootstrap."""
    all_preds, all_gts = [], []
    for ds in datasets:
        if ds["name"] not in sources:
            continue
        cfg = ds.get(task_key)
        if not cfg:
            continue
        all_preds.extend(load_jsonl(str(cfg["pred"])))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            all_gts.extend(json.load(f))
    if not all_preds:
        return None, None

    pred_by_qid = {p["question_id"]: p for p in all_preds}
    raw = defaultdict(lambda: defaultdict(lambda: {"preds": [], "gt": None}))
    for gt in all_gts:
        qid = gt["question_id"]
        pid = _extract_patient_id_qa(gt)
        if patient_filter and pid not in patient_filter:
            continue
        if qid not in pred_by_qid:
            continue
        qtype = gt["Question_type"]
        raw[pid][qtype]["preds"].append(pred_by_qid[qid]["text"])
        if raw[pid][qtype]["gt"] is None:
            raw[pid][qtype]["gt"] = gt["label"]

    patient_data = {}
    for pid, qtypes in raw.items():
        patient_data[pid] = {}
        for qtype, d in qtypes.items():
            patient_data[pid][qtype] = (majority_vote(d["preds"]), d["gt"])

    patient_ids = sorted(patient_data.keys())

    def get_metrics(sampled_pids):
        all_p, all_g = [], []
        by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
        for pid in sampled_pids:
            if pid not in patient_data:
                continue
            for qtype, (vpred, vgt) in patient_data[pid].items():
                by_qtype[qtype]["pred"].append(vpred)
                by_qtype[qtype]["gt"].append(vgt)
                all_p.append(vpred)
                all_g.append(vgt)
        metrics = {}
        for qtype, d in by_qtype.items():
            if d["gt"]:
                metrics[f"{qtype}_acc"] = sum(
                    1 for a, b in zip(d["pred"], d["gt"]) if a == b
                ) / len(d["gt"])
                metrics[f"{qtype}_f1"] = compute_f1(d["pred"], d["gt"])
        if all_g:
            metrics["Overall_acc"] = sum(
                1 for a, b in zip(all_p, all_g) if a == b
            ) / len(all_g)
            metrics["Overall_f1"] = compute_f1(all_p, all_g)
        return metrics

    return patient_ids, get_metrics


def prepare_merged_qa_single(sources, datasets, task_key, patient_filter=None):
    """Merge QA data from multiple source datasets for single-modal bootstrap."""
    all_pairs = []
    for ds in datasets:
        if ds["name"] not in sources:
            continue
        cfg = ds.get(task_key)
        if not cfg:
            continue
        src_preds = load_jsonl(str(cfg["pred"]))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            src_gts = json.load(f)
        for pred, gt in zip(src_preds, src_gts):
            all_pairs.append((pred, gt))
    if not all_pairs:
        return None, None

    raw = defaultdict(lambda: {"pred": [], "gt": []})
    for pred, gt in all_pairs:
        pid = _extract_patient_id_single(gt)
        if patient_filter and pid not in patient_filter:
            continue
        mod = gt.get("modality", "unknown")
        qtype = gt["Question_type"]
        raw[(mod, pid, qtype)]["pred"].append(pred["text"])
        raw[(mod, pid, qtype)]["gt"].append(gt["label"])

    patient_data = defaultdict(dict)
    for (mod, pid, qtype), data in raw.items():
        patient_data[pid][(mod, qtype)] = (data["pred"][0], data["gt"][0])

    patient_ids = sorted(patient_data.keys())

    def get_metrics(sampled_pids):
        by_mod_qtype = defaultdict(lambda: {"pred": [], "gt": []})
        for pid in sampled_pids:
            if pid not in patient_data:
                continue
            for (mod, qtype), (vpred, vgt) in patient_data[pid].items():
                by_mod_qtype[(mod, qtype)]["pred"].append(vpred)
                by_mod_qtype[(mod, qtype)]["gt"].append(vgt)
        metrics = {}
        modalities = sorted(set(m for (m, _) in by_mod_qtype.keys()))
        for mod in modalities:
            mod_p, mod_g = [], []
            for qtype in sorted(set(q for (m, q) in by_mod_qtype.keys() if m == mod)):
                d = by_mod_qtype[(mod, qtype)]
                if d["gt"]:
                    metrics[f"{mod}_{qtype}_acc"] = sum(
                        1 for a, b in zip(d["pred"], d["gt"]) if a == b
                    ) / len(d["gt"])
                    metrics[f"{mod}_{qtype}_f1"] = compute_f1(d["pred"], d["gt"])
                    mod_p.extend(d["pred"])
                    mod_g.extend(d["gt"])
            if mod_g:
                metrics[f"{mod}_Overall_acc"] = sum(
                    1 for a, b in zip(mod_p, mod_g) if a == b
                ) / len(mod_g)
                metrics[f"{mod}_Overall_f1"] = compute_f1(mod_p, mod_g)
        all_p, all_g = [], []
        all_by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
        for (mod, qtype), d in by_mod_qtype.items():
            all_by_qtype[qtype]["pred"].extend(d["pred"])
            all_by_qtype[qtype]["gt"].extend(d["gt"])
            all_p.extend(d["pred"])
            all_g.extend(d["gt"])
        for qtype, d in all_by_qtype.items():
            if d["gt"]:
                metrics[f"ALL_{qtype}_acc"] = sum(
                    1 for a, b in zip(d["pred"], d["gt"]) if a == b
                ) / len(d["gt"])
                metrics[f"ALL_{qtype}_f1"] = compute_f1(d["pred"], d["gt"])
        if all_g:
            metrics["ALL_Overall_acc"] = sum(
                1 for a, b in zip(all_p, all_g) if a == b
            ) / len(all_g)
            metrics["ALL_Overall_f1"] = compute_f1(all_p, all_g)
        return metrics

    return patient_ids, get_metrics


def prepare_merged_report(sources, datasets, task_key, evaluators, mode,
                          patient_filter=None):
    """Merge report data from multiple source datasets.

    Pre-computes per-patient scores once for fast bootstrap resampling.
    """
    all_preds, all_gts = [], []
    for ds in datasets:
        if ds["name"] not in sources:
            continue
        cfg = ds.get(task_key)
        if not cfg:
            continue
        all_preds.extend(load_jsonl(str(cfg["pred"])))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            all_gts.extend(json.load(f))
    if not all_preds:
        return None, None

    gt_mapping = {gt["question_id"]: gt for gt in all_gts}
    bleu_eval, cider_eval, rouge_eval = evaluators

    if mode == "multi":
        data_by_patient = defaultdict(lambda: {"preds": [], "gts": []})
        for pred in all_preds:
            qid = pred["question_id"]
            if qid not in gt_mapping:
                continue
            pid = extract_patient_id_report(qid)
            if patient_filter and str(pid) not in patient_filter:
                continue
            data_by_patient[pid]["preds"].append(pred["text"])
            data_by_patient[pid]["gts"].append(gt_mapping[qid]["label"])

        patient_report = {}
        for pid, data in data_by_patient.items():
            patient_report[pid] = (data["gts"][0], data["preds"][0])

        patient_ids = sorted(patient_report.keys())

        # Pre-compute per-patient scores (run evaluators ONCE)
        gt_dict = {pid: [gt] for pid, (gt, _) in patient_report.items()}
        pred_dict = {pid: [pred] for pid, (_, pred) in patient_report.items()}
        per_sample = _precompute_report_per_sample(
            gt_dict, pred_dict, bleu_eval, cider_eval, rouge_eval)

        score_matrix, pid_to_idx = _build_score_matrix(
            patient_ids, per_sample, _REPORT_METRICS)
        get_metrics = _make_fast_get_metrics(
            score_matrix, pid_to_idx, _REPORT_METRICS)

        return patient_ids, get_metrics
    else:
        # Single-modal merged
        data_by_mod_patient = defaultdict(lambda: {"preds": [], "gts": []})
        for pred in all_preds:
            qid = pred["question_id"]
            if qid not in gt_mapping:
                continue
            gt = gt_mapping[qid]
            pid = _extract_patient_id_single(gt)
            if patient_filter and pid not in patient_filter:
                continue
            mod = gt.get("modality", "unknown")
            data_by_mod_patient[(mod, pid)]["preds"].append(pred["text"])
            data_by_mod_patient[(mod, pid)]["gts"].append(gt["label"])

        patient_report_by_mod = defaultdict(dict)
        for (mod, pid), data in data_by_mod_patient.items():
            patient_report_by_mod[pid][mod] = (data["gts"][0], data["preds"][0])

        patient_ids = sorted(patient_report_by_mod.keys())

        # Pre-compute per-patient scores for each modality and ALL
        combined_scores = {pid: {} for pid in patient_ids}

        modalities = sorted(set(m for pid in patient_ids
                                for m in patient_report_by_mod[pid].keys()))
        for mod in modalities:
            gt_dict, pred_dict = {}, {}
            for pid in patient_ids:
                if mod in patient_report_by_mod[pid]:
                    gt_t, pr_t = patient_report_by_mod[pid][mod]
                    gt_dict[pid] = [gt_t]
                    pred_dict[pid] = [pr_t]
            if gt_dict:
                per_sample = _precompute_report_per_sample(
                    gt_dict, pred_dict, bleu_eval, cider_eval, rouge_eval)
                for pid, scores in per_sample.items():
                    for m, v in scores.items():
                        combined_scores[pid][f"{mod}_{m}"] = v

        # ALL: use first modality entry per patient
        all_gt_dict, all_pred_dict = {}, {}
        for pid in patient_ids:
            first_mod = next(iter(patient_report_by_mod[pid]))
            gt_t, pr_t = patient_report_by_mod[pid][first_mod]
            all_gt_dict[pid] = [gt_t]
            all_pred_dict[pid] = [pr_t]
        if all_gt_dict:
            per_sample = _precompute_report_per_sample(
                all_gt_dict, all_pred_dict, bleu_eval, cider_eval, rouge_eval)
            for pid, scores in per_sample.items():
                for m, v in scores.items():
                    combined_scores[pid][f"ALL_{m}"] = v

        all_metric_names = sorted(set(
            k for pid in patient_ids for k in combined_scores[pid].keys()))
        score_matrix, pid_to_idx = _build_score_matrix(
            patient_ids, combined_scores, all_metric_names)
        get_metrics = _make_fast_get_metrics(
            score_matrix, pid_to_idx, all_metric_names)

        return patient_ids, get_metrics


def prepare_merged_grounding_single(sources, datasets, patient_filter=None):
    """Merge grounding data from multiple source datasets."""
    all_preds_raw, all_gt_data = [], []
    for ds in datasets:
        if ds["name"] not in sources:
            continue
        cfg = ds.get("grounding")
        if not cfg:
            continue
        all_preds_raw.extend(load_jsonl(str(cfg["pred"])))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            all_gt_data.extend(json.load(f))
    if not all_preds_raw:
        return None, None

    pred_map = {p["question_id"]: p["text"] for p in all_preds_raw}
    patient_slices = defaultdict(list)
    iou_thresholds = [0.1, 0.3, 0.5]

    for gt_entry in all_gt_data:
        qid = gt_entry["question_id"]
        if qid not in pred_map:
            continue
        pid = _extract_patient_id_single(gt_entry)
        if patient_filter and pid not in patient_filter:
            continue
        pred_box = parse_bounding_box_single(pred_map[qid])
        gt_box = parse_bounding_box_single(gt_entry["label"])
        if pred_box is None or gt_box is None:
            continue
        iou = calculate_iou(pred_box, gt_box)
        if iou is None:
            continue
        mod = gt_entry.get("modality", "unknown")
        patient_slices[pid].append((iou, mod))

    patient_ids = sorted(patient_slices.keys())

    def get_metrics(sampled_pids):
        all_ious = []
        by_mod = defaultdict(list)
        for pid in sampled_pids:
            if pid not in patient_slices:
                continue
            for iou, mod in patient_slices[pid]:
                all_ious.append(iou)
                by_mod[mod].append(iou)
        metrics = {}
        for mod, ious in by_mod.items():
            arr = np.array(ious)
            metrics[f"{mod}_mIoU"] = float(np.mean(arr))
            for t in iou_thresholds:
                metrics[f"{mod}_IoU@{t}"] = float(np.mean(arr >= t))
        if all_ious:
            arr = np.array(all_ious)
            metrics["ALL_mIoU"] = float(np.mean(arr))
            for t in iou_thresholds:
                metrics[f"ALL_IoU@{t}"] = float(np.mean(arr >= t))
        return metrics

    return patient_ids, get_metrics


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def fmt_ci(val):
    """Format a CI result as 'point (lower-upper) mean±std'."""
    return f"{val['point']:.4f} ({val['ci_lower']:.4f}-{val['ci_upper']:.4f}) mean={val['mean']:.4f}±{val['std']:.4f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def files_exist(cfg):
    return cfg and Path(cfg["gt"]).exists() and Path(cfg["pred"]).exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CI for evaluation metrics.")
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--n-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: evaluation_archive/results_bootstrap_{mode})")
    parser.add_argument("--pred-subdir", default=None,
                        help="Replace 'pred' subdir in prediction paths "
                             "(e.g. --pred-subdir pred_zeroshot)")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = f"_{args.pred_subdir}" if args.pred_subdir else ""
        args.output_dir = f"evaluation_archive/results_bootstrap_{args.mode}{suffix}"

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = MULTI_DATASETS if args.mode == "multi" else SINGLE_DATASETS
    MERGED = MULTI_MERGED if args.mode == "multi" else SINGLE_MERGED

    # Remap prediction paths if --pred-subdir is specified
    if args.pred_subdir:
        import copy
        DATASETS = copy.deepcopy(DATASETS)
        MERGED = copy.deepcopy(MERGED)
        for ds in DATASETS:
            for task_key in ["qa", "qa_bbox", "report", "report_bbox", "grounding"]:
                if task_key in ds and ds[task_key]:
                    old = str(ds[task_key]["pred"])
                    ds[task_key]["pred"] = Path(
                        old.replace("/pred/", f"/{args.pred_subdir}/"))
        print(f"[INFO] Using pred subdir: {args.pred_subdir}")

    # Load report evaluators
    try:
        evaluators = _load_report_evaluators()
        has_report = True
    except ImportError:
        has_report = False
        evaluators = None
        print("[WARNING] pycocoevalcap not available, skipping report bootstrap.")

    n_iter = args.n_iterations
    seed = args.seed
    all_rows = []  # (dataset, task, metric_key, point, ci_lower, ci_upper, n)

    # ----- Regular datasets -----
    for ds in DATASETS:
        name = ds["name"]
        pf = ds.get("patient_filter")
        print(f"\n{'='*60}")
        print(f"Bootstrap: {name}" + (f" (filtered)" if pf else ""))
        print(f"{'='*60}")

        # QA tasks
        for task_key, task_label in [("qa", "QA"), ("qa_bbox", "QA+bbox")]:
            cfg = ds.get(task_key)
            if not files_exist(cfg):
                print(f"  {task_label}: SKIPPED")
                continue
            print(f"  {task_label} ...")
            if args.mode == "multi":
                pids, mfn = prepare_qa_multi(cfg["gt"], cfg["pred"], pf)
            else:
                pids, mfn = prepare_qa_single(cfg["gt"], cfg["pred"], pf)
            ci = bootstrap_ci(pids, mfn, n_iter=n_iter, seed=seed,
                              label=f"{name}/{task_label}")
            for metric_name, vals in ci.items():
                all_rows.append([name, task_label, metric_name,
                                 f"{vals['point']:.4f}",
                                 f"{vals['ci_lower']:.4f}",
                                 f"{vals['ci_upper']:.4f}",
                                 f"{vals['mean']:.4f}",
                                 f"{vals['std']:.4f}",
                                 len(pids),
                                 fmt_ci(vals)])
            print(f"    {len(pids)} patients, {len(ci)} metrics")

        # Report tasks
        if has_report:
            for task_key, task_label in [("report", "Report"),
                                         ("report_bbox", "Report+bbox")]:
                cfg = ds.get(task_key)
                if not files_exist(cfg):
                    print(f"  {task_label}: SKIPPED")
                    continue
                print(f"  {task_label} ...")
                if args.mode == "multi":
                    pids, mfn = prepare_report_multi(
                        cfg["gt"], cfg["pred"], evaluators, pf)
                else:
                    pids, mfn = prepare_report_single(
                        cfg["gt"], cfg["pred"], evaluators, pf)
                ci = bootstrap_ci(pids, mfn, n_iter=n_iter, seed=seed,
                                  label=f"{name}/{task_label}")
                for metric_name, vals in ci.items():
                    all_rows.append([name, task_label, metric_name,
                                     f"{vals['point']:.4f}",
                                     f"{vals['ci_lower']:.4f}",
                                     f"{vals['ci_upper']:.4f}",
                                     f"{vals['mean']:.4f}",
                                     f"{vals['std']:.4f}",
                                     len(pids),
                                     fmt_ci(vals)])
                print(f"    {len(pids)} patients, {len(ci)} metrics")

        # Grounding (single only)
        if args.mode == "single":
            cfg = ds.get("grounding")
            if files_exist(cfg):
                print(f"  Grounding ...")
                pids, mfn = prepare_grounding_single(
                    cfg["gt"], cfg["pred"], pf)
                ci = bootstrap_ci(pids, mfn, n_iter=n_iter, seed=seed,
                                  label=f"{name}/Grounding")
                for metric_name, vals in ci.items():
                    all_rows.append([name, "Grounding", metric_name,
                                     f"{vals['point']:.4f}",
                                     f"{vals['ci_lower']:.4f}",
                                     f"{vals['ci_upper']:.4f}",
                                     f"{vals['mean']:.4f}",
                                     f"{vals['std']:.4f}",
                                     len(pids),
                                     fmt_ci(vals)])
                print(f"    {len(pids)} patients, {len(ci)} metrics")
            else:
                print(f"  Grounding: SKIPPED")

    # ----- Merged datasets -----
    for merged in MERGED:
        mname = merged["name"]
        sources = merged["sources"]
        print(f"\n{'='*60}")
        print(f"Bootstrap merged: {mname}")
        print(f"{'='*60}")

        for task_key, task_label in [("qa", "QA"), ("qa_bbox", "QA+bbox")]:
            if args.mode == "multi":
                pids, mfn = prepare_merged_qa_multi(sources, DATASETS, task_key)
            else:
                pids, mfn = prepare_merged_qa_single(sources, DATASETS, task_key)
            if pids is None:
                print(f"  {task_label}: SKIPPED")
                continue
            print(f"  {task_label} ...")
            ci = bootstrap_ci(pids, mfn, n_iter=n_iter, seed=seed,
                              label=f"{mname}/{task_label}")
            for metric_name, vals in ci.items():
                all_rows.append([mname, task_label, metric_name,
                                 f"{vals['point']:.4f}",
                                 f"{vals['ci_lower']:.4f}",
                                 f"{vals['ci_upper']:.4f}",
                                 f"{vals['mean']:.4f}",
                                 f"{vals['std']:.4f}",
                                 len(pids),
                                 fmt_ci(vals)])
            print(f"    {len(pids)} patients, {len(ci)} metrics")

        if has_report:
            for task_key, task_label in [("report", "Report"),
                                         ("report_bbox", "Report+bbox")]:
                pids, mfn = prepare_merged_report(
                    sources, DATASETS, task_key, evaluators, args.mode)
                if pids is None:
                    print(f"  {task_label}: SKIPPED")
                    continue
                print(f"  {task_label} ...")
                ci = bootstrap_ci(pids, mfn, n_iter=n_iter, seed=seed,
                                  label=f"{mname}/{task_label}")
                for metric_name, vals in ci.items():
                    all_rows.append([mname, task_label, metric_name,
                                     f"{vals['point']:.4f}",
                                     f"{vals['ci_lower']:.4f}",
                                     f"{vals['ci_upper']:.4f}",
                                     f"{vals['mean']:.4f}",
                                     f"{vals['std']:.4f}",
                                     len(pids),
                                     fmt_ci(vals)])
                print(f"    {len(pids)} patients, {len(ci)} metrics")

        if args.mode == "single":
            pids, mfn = prepare_merged_grounding_single(sources, DATASETS)
            if pids is not None:
                print(f"  Grounding ...")
                ci = bootstrap_ci(pids, mfn, n_iter=n_iter, seed=seed,
                                  label=f"{mname}/Grounding")
                for metric_name, vals in ci.items():
                    all_rows.append([mname, "Grounding", metric_name,
                                     f"{vals['point']:.4f}",
                                     f"{vals['ci_lower']:.4f}",
                                     f"{vals['ci_upper']:.4f}",
                                     f"{vals['mean']:.4f}",
                                     f"{vals['std']:.4f}",
                                     len(pids),
                                     fmt_ci(vals)])
                print(f"    {len(pids)} patients, {len(ci)} metrics")

    # ----- Write output -----
    csv_path = output_dir / "bootstrap_ci.csv"
    write_csv(csv_path,
              ["dataset", "task", "metric", "point", "ci_lower", "ci_upper",
               "mean", "std", "n_patients", "formatted"],
              all_rows)
    print(f"\n\nCSV written to: {csv_path}")

    # ----- Print summary markdown tables -----
    print(f"\n{'='*80}")
    print(f"Bootstrap 95% CI Summary ({args.mode}-modal, {n_iter} iterations)")
    print(f"{'='*80}")

    # Group rows by task
    rows_by_task = defaultdict(list)
    for row in all_rows:
        rows_by_task[row[1]].append(row)

    for task_label in ["QA", "QA+bbox", "Report", "Report+bbox", "Grounding"]:
        task_rows = rows_by_task.get(task_label, [])
        if not task_rows:
            continue
        print(f"\n### {task_label}")
        print("| Dataset | Metric | Point | Mean±Std | 95% CI | N |")
        print("|---|---|---|---|---|---|")
        for row in task_rows:
            ds, _, metric, point, lo, hi, mean, std, n, formatted = row
            print(f"| {ds} | {metric} | {point} | {mean}±{std} | {lo}-{hi} | {n} |")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
