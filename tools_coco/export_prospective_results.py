#!/usr/bin/env python3
"""
Export per-patient results for the updated 100-patient prospective GXMU cohort.

For each patient, creates a folder containing:
  images/        - ALL MRI images across all voting samples (PRE, AP, PVP, T2WI)
  grounding/     - grounding images with bbox drawn for all predicted slices (single-modal)
  qa.txt         - QA1–QA4 majority-voted prediction vs GT (correct/wrong)
  report.txt     - report generation GT + prediction

Usage:
    python tools_coco/export_prospective_results.py
    python tools_coco/export_prospective_results.py --output my_output_dir
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Patient cohort (100 patients: original minus 13,28,40 plus 12,24,81)
# ---------------------------------------------------------------------------
NEW_IDS = {
    '2', '6', '7', '8', '9', '10', '11', '12', '15', '16', '17', '19', '20',
    '21', '22', '23', '24', '26', '27', '31', '32', '33', '35', '37', '39',
    '41', '43', '44', '45', '46', '48', '50', '51', '52', '54', '55', '56',
    '58', '59', '62', '63', '64', '68', '69', '70', '72', '73', '74', '76',
    '77', '78', '79', '80', '81', '82', '83', '84', '86', '88', '92', '93',
    '95', '99', '100', '103', '104', '105', '106', '107', '108', '109', '110',
    '112', '113', '114', '115', '116', '119', '120', '121', '122', '123',
    '124', '126', '127', '128', '129', '130', '131', '132', '133', '134',
    '135', '136', '137', '138', '140', '142', '143', '147',
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE = BASE_DIR / "evaluation_archive"
IMG_ROOT = Path("/mnt/data/by/data/coco_new/data_external_batch2")
DATASET = "liver_MRI_EXTERNAL_prospective_GXMU_ready_hcc_icc_mask_64"

QA_GT_FILE = ARCHIVE / "multi_gating/prospective_gxmu/gt/test_QA_EN_multimodal_4mod_with_pred_bbox.json"
QA_PRED_FILE = ARCHIVE / "multi_gating/prospective_gxmu/pred/hcc_icc_mask_64_multi_qa_with_pred_bbox.json"
REPORT_GT_FILE = ARCHIVE / "multi_gating/prospective_gxmu/gt/test_report_generation_EN_multimodal.json"
REPORT_PRED_FILE = ARCHIVE / "multi_gating/prospective_gxmu/pred/hcc_icc_mask_64_multi_report.json"
GROUNDING_GT_FILE = ARCHIVE / "single/prospective_gxmu/gt/test_bbox.json"
GROUNDING_PRED_FILE = ARCHIVE / "single/prospective_gxmu/pred/hcc_icc_mask_64_grounding.json"

MODALITIES = ["PRE", "AP", "PVP", "T2WI"]

# Short question labels for qa.txt header
QA_LABELS = {
    "QA1": "Does this patient have a focal liver lesion?",
    "QA2": "Does this patient have a malignant tumor in the liver?",
    "QA3": "What is the primary liver tumor subtype?",
    "QA4": "What is the most likely diagnosis for this lesion?",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]


def parse_bbox(text):
    """Parse [[x1, y1, x2, y2]] from <box> text. Coords are 0-1000 normalized."""
    m = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
    return list(map(int, m.groups())) if m else None


def scale_bbox(coords, W, H):
    """Scale 0-1000 normalized coords to pixel coords."""
    return [
        int(coords[0] / 1000 * W),
        int(coords[1] / 1000 * H),
        int(coords[2] / 1000 * W),
        int(coords[3] / 1000 * H),
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    # --- QA GT: all entries grouped by patient, then by question type ---
    with open(QA_GT_FILE) as f:
        qa_gt_all = json.load(f)

    # qa_gt_by_pid[pid][qtype] = list of GT entries
    qa_gt_by_pid = defaultdict(lambda: defaultdict(list))
    for entry in qa_gt_all:
        pid = entry["patient_prefix"]
        if pid in NEW_IDS:
            qa_gt_by_pid[pid][entry["Question_type"]].append(entry)

    # --- QA Pred: keyed by question_id ---
    qa_pred_by_qid = {p["question_id"]: p["text"] for p in load_jsonl(QA_PRED_FILE)}

    # --- Report GT: first entry per patient ---
    with open(REPORT_GT_FILE) as f:
        report_gt_all = json.load(f)

    report_gt_by_pid = {}
    for entry in report_gt_all:
        pid = entry["patient_prefix"]
        if pid in NEW_IDS and pid not in report_gt_by_pid:
            report_gt_by_pid[pid] = entry

    # --- Report Pred: keyed by question_id ---
    report_pred_by_qid = {p["question_id"]: p["text"] for p in load_jsonl(REPORT_PRED_FILE)}

    # --- Grounding GT: gt_bbox[pid][mod][slice_str] = coords ---
    with open(GROUNDING_GT_FILE) as f:
        grounding_gt_all = json.load(f)

    gt_bbox = defaultdict(lambda: defaultdict(dict))
    for entry in grounding_gt_all:
        parts = entry["question_id"].split("/")
        if len(parts) == 3:
            pid, mod, slc = parts
            coords = parse_bbox(entry["label"])
            if coords:
                gt_bbox[pid][mod][slc] = coords

    # --- Grounding Pred: pred_bbox[pid][mod][slice_str] = coords ---
    pred_bbox = defaultdict(lambda: defaultdict(dict))
    for entry in load_jsonl(GROUNDING_PRED_FILE):
        parts = entry["question_id"].split("/")
        if len(parts) == 3:
            pid, mod, slc = parts
            coords = parse_bbox(entry["text"])
            if coords:
                pred_bbox[pid][mod][slc] = coords

    return (
        qa_gt_by_pid, qa_pred_by_qid,
        report_gt_by_pid, report_pred_by_qid,
        gt_bbox, pred_bbox,
    )


# ---------------------------------------------------------------------------
# Per-patient export
# ---------------------------------------------------------------------------

def export_patient(pid, out_dir, qa_gt_by_pid, qa_pred_by_qid,
                   report_gt_by_pid, report_pred_by_qid,
                   gt_bbox, pred_bbox):
    pat_dir = out_dir / f"patient_{int(pid):03d}"
    img_dir = pat_dir / "images"

    # --- Collect all unique (mod, slice) from ALL QA entries for this patient ---
    all_slices = defaultdict(set)  # mod -> set of slice_str
    for qtype_entries in qa_gt_by_pid[pid].values():
        for entry in qtype_entries:
            for mod, slc in entry.get("slice_nums", {}).items():
                all_slices[mod].add(str(slc))

    # --- Copy images into per-modality subfolders ---
    for mod in MODALITIES:
        mod_img_dir = img_dir / mod
        mod_img_dir.mkdir(parents=True, exist_ok=True)
        for slc in sorted(all_slices[mod], key=int):
            src = IMG_ROOT / DATASET / pid / mod / "images" / f"{slc}.png"
            dst = mod_img_dir / f"{int(slc):03d}.png"
            if src.exists():
                shutil.copy(src, dst)

    # --- QA1–QA4: majority vote per question type ---
    qa_lines = []
    for qtype in ["QA1", "QA2", "QA3", "QA4"]:
        entries = qa_gt_by_pid[pid].get(qtype, [])
        if not entries:
            continue

        gt_label = entries[0]["label"]

        # Collect predictions across all slices for this question type
        preds = []
        for entry in entries:
            qid = entry["question_id"]
            pred_text = qa_pred_by_qid.get(qid)
            if pred_text:
                preds.append(pred_text)

        if preds:
            voted_pred = majority_vote(preds)
            correct_str = "CORRECT" if voted_pred == gt_label else "WRONG"
        else:
            voted_pred = "N/A"
            correct_str = "N/A"

        qa_lines.append(f"[{qtype}] {QA_LABELS.get(qtype, qtype)}")
        qa_lines.append(f"GT:   {gt_label}")
        qa_lines.append(f"Pred: {voted_pred}   [{correct_str}]")
        qa_lines.append("")

    (pat_dir / "qa.txt").write_text("\n".join(qa_lines), encoding="utf-8")

    # --- Report ---
    report_gt_entry = report_gt_by_pid.get(pid)
    if report_gt_entry:
        report_qid = report_gt_entry["question_id"]
        report_gt_text = report_gt_entry["label"]
        report_pred_text = report_pred_by_qid.get(report_qid, "N/A")
    else:
        report_gt_text = "N/A"
        report_pred_text = "N/A"

    report_lines = [
        "=== Ground Truth ===",
        report_gt_text,
        "",
        "=== Prediction ===",
        report_pred_text,
    ]
    (pat_dir / "report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # --- Grounding: all slices with predictions ---
    has_grounding = False
    for mod in pred_bbox[pid]:
        for slc in sorted(pred_bbox[pid][mod], key=int):
            pred_coords = pred_bbox[pid][mod][slc]
            gt_coords = gt_bbox[pid][mod].get(slc)

            src_img = IMG_ROOT / DATASET / pid / mod / "images" / f"{slc}.png"
            if not src_img.exists():
                continue

            has_grounding = True
            slc_int = int(slc)
            grd_gt_dir = pat_dir / "grounding_gt" / mod
            grd_pred_dir = pat_dir / "grounding_pred" / mod
            grd_both_dir = pat_dir / "grounding_both" / mod
            grd_gt_dir.mkdir(parents=True, exist_ok=True)
            grd_pred_dir.mkdir(parents=True, exist_ok=True)
            grd_both_dir.mkdir(parents=True, exist_ok=True)

            img_base = Image.open(src_img).convert("RGB")
            W, H = img_base.size

            # Image 1: GT only (red)
            img0 = img_base.copy()
            draw0 = ImageDraw.Draw(img0)
            if gt_coords:
                draw0.rectangle(scale_bbox(gt_coords, W, H), outline="red", width=2)
            img0.save(grd_gt_dir / f"{slc_int:03d}.png")

            # Image 2: pred only (blue)
            img1 = img_base.copy()
            draw1 = ImageDraw.Draw(img1)
            if pred_coords:
                draw1.rectangle(scale_bbox(pred_coords, W, H), outline="blue", width=2)
            img1.save(grd_pred_dir / f"{slc_int:03d}.png")

            # Image 3: GT (red) + pred (blue)
            img2 = img_base.copy()
            draw2 = ImageDraw.Draw(img2)
            if gt_coords:
                draw2.rectangle(scale_bbox(gt_coords, W, H), outline="red", width=2)
            if pred_coords:
                draw2.rectangle(scale_bbox(pred_coords, W, H), outline="blue", width=2)
            img2.save(grd_both_dir / f"{slc_int:03d}.png")

    # For patients without grounding, copy images/ as grounding_pred/
    if not has_grounding:
        shutil.copytree(img_dir, pat_dir / "grounding_pred")

    return has_grounding


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export prospective 100-patient results")
    parser.add_argument("--output", default="prospective_100_results",
                        help="Output directory (default: prospective_100_results)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data files...")
    (qa_gt_by_pid, qa_pred_by_qid,
     report_gt_by_pid, report_pred_by_qid,
     gt_bbox, pred_bbox) = load_all_data()

    print(f"Exporting {len(NEW_IDS)} patients to {out_dir}/")
    n_with_grounding = 0
    n_without_grounding = 0

    for pid in sorted(NEW_IDS, key=int):
        has_grounding = export_patient(
            pid, out_dir,
            qa_gt_by_pid, qa_pred_by_qid,
            report_gt_by_pid, report_pred_by_qid,
            gt_bbox, pred_bbox,
        )
        if has_grounding:
            n_with_grounding += 1
        else:
            n_without_grounding += 1

    # --- Save GT bbox values as CSV ---
    bbox_rows = []
    for pid_str in sorted(NEW_IDS, key=int):
        for mod in MODALITIES:
            for slc in sorted(gt_bbox[pid_str][mod], key=int):
                coords = gt_bbox[pid_str][mod][slc]
                bbox_rows.append({
                    'patient_id': int(pid_str),
                    'modality': mod,
                    'slice': int(slc),
                    'x1': coords[0], 'y1': coords[1],
                    'x2': coords[2], 'y2': coords[3],
                })
    bbox_csv = out_dir / "gt_bbox.csv"
    with open(bbox_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'modality', 'slice',
                                                'x1', 'y1', 'x2', 'y2'])
        writer.writeheader()
        writer.writerows(bbox_rows)
    n_bbox_patients = len({r['patient_id'] for r in bbox_rows})
    print(f"  GT bbox CSV:       {len(bbox_rows)} entries from {n_bbox_patients} patients -> {bbox_csv}")

    print(f"\nDone. Exported {len(NEW_IDS)} patients to {out_dir}/")
    print(f"  With grounding:    {n_with_grounding} patients")
    print(f"  Without grounding: {n_without_grounding} patients (Benign/No focal)")


if __name__ == "__main__":
    main()
