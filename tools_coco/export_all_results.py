#!/usr/bin/env python3
"""
Export per-patient results for ALL internal + external databases.
Same folder structure as prospective export. Databases without bbox skip grounding.

Usage:
    python tools_coco/export_all_results.py
    python tools_coco/export_all_results.py --output groundliver_results
    python tools_coco/export_all_results.py --db external_enshi internal
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE = BASE_DIR / "evaluation_archive"
DATA_ROOT = Path("/mnt/data/by/data/coco_new")
MODALITIES = ["PRE", "AP", "PVP", "T2WI"]

QA_LABELS = {
    "QA1": "Does this patient have a focal liver lesion?",
    "QA2": "Does this patient have a malignant tumor in the liver?",
    "QA3": "What is the primary liver tumor subtype?",
    "QA4": "What is the most likely diagnosis for this lesion?",
}

# ---------------------------------------------------------------------------
# Database configurations
# ---------------------------------------------------------------------------
DB_CONFIG = {
    'external1_gxmu_hcc_icc': {
        'img_root': 'data_external_final_pre',
        'dataset': 'liver_MRI_EXTERNAL1_HCC',
        'qa_gt': 'test_QA_EN_external_multimodal_4mod_with_bbox.json',
        'qa_pred': 'external1_multi_qa_with_bbox_gating.json',
        'report_gt': 'test_report_generation_EN_multimodal_with_bbox.json',
        'report_pred': 'external1_multi_report_with_bbox_gating.json',
        'has_bbox': True,
        'grounding_pred': 'external1_single_grounding.json',
    },
    'external_enshi': {
        'img_root': 'data_external_final',
        'dataset': 'liver_MRI_EXTERNAL2_ENSHI_ready',
        'qa_pred': 'external_ENSHI_multi_qa_with_bbox_gating.json',
        'report_pred': 'external_ENSHI_multi_report_with_bbox_gating.json',
        'has_bbox': True,
        'grounding_pred': 'external_ENSHI_single_grounding.json',
    },
    'external_sanya': {
        'img_root': 'data_external_final',
        'dataset': 'liver_MRI_EXTERNAL3_SanYa_ready',
        'qa_pred': 'external_SanYa_multi_qa_with_bbox_gating.json',
        'report_pred': 'external_SanYa_multi_report_with_bbox_gating.json',
        'has_bbox': True,
        'grounding_pred': 'external_SanYa_single_grounding.json',
    },
    'external_nanning': {
        'img_root': 'data_external_batch2',
        'dataset': 'liver_MRI_EXTERNAL4_Nanning_ready',
        'qa_pred': 'external_batch2_multi_liver_MRI_EXTERNAL4_Nanning_ready_multimodal_qa_with_bbox.json',
        'report_pred': 'external_batch2_multi_liver_MRI_EXTERNAL4_Nanning_ready_multimodal_report_with_bbox.json',
        'has_bbox': False,
    },
    'external_beihai': {
        'img_root': 'data_external_batch2',
        'dataset': 'liver_MRI_EXTERNAL5_Beihai_First_ready',
        'qa_pred': 'external_batch2_multi_liver_MRI_EXTERNAL5_Beihai_First_ready_multimodal_qa_with_bbox.json',
        'report_pred': 'external_batch2_multi_liver_MRI_EXTERNAL5_Beihai_First_ready_multimodal_report_with_bbox.json',
        'has_bbox': False,
    },
    'external_guigang': {
        'img_root': 'data_external_batch2',
        'dataset': 'liver_MRI_EXTERNAL6_Guigang_ready',
        'qa_pred': 'external_batch2_multi_liver_MRI_EXTERNAL6_Guigang_ready_multimodal_qa_with_bbox.json',
        'report_pred': 'external_batch2_multi_liver_MRI_EXTERNAL6_Guigang_ready_multimodal_report_with_bbox.json',
        'has_bbox': False,
    },
    'external_guilin': {
        'img_root': 'data_external_batch2',
        'dataset': 'liver_MRI_EXTERNAL7_Guilin_ready',
        'qa_pred': 'external_batch2_multi_liver_MRI_EXTERNAL7_Guilin_ready_multimodal_qa_with_bbox.json',
        'report_pred': 'external_batch2_multi_liver_MRI_EXTERNAL7_Guilin_ready_multimodal_report_with_bbox.json',
        'has_bbox': False,
    },
    'external_henan': {
        'img_root': 'data_external_batch2',
        'dataset': 'liver_MRI_EXTERNAL8_HeNan_ready',
        'qa_pred': 'external_batch2_multi_liver_MRI_EXTERNAL8_HeNan_ready_multimodal_qa_with_bbox.json',
        'report_pred': 'external_batch2_multi_liver_MRI_EXTERNAL8_HeNan_ready_multimodal_report_with_bbox.json',
        'has_bbox': False,
    },
    'external_gxmu_benign_normal': {
        'img_root': 'data_external_final',
        'dataset': 'liver_MRI_EXTERNAL1_GXMU_ready_benign_normal',
        'qa_pred': 'external_GXMU_bn_multi_qa_with_bbox_gating.json',
        'report_pred': 'external_GXMU_bn_multi_report_with_bbox_gating.json',
        'has_bbox': False,
    },
    'internal': {
        'img_root': 'data',
        'dataset': '',
        'qa_pred': 'internal_multi_qa_with_bbox_gating.json',
        'report_pred': 'internal_multi_report_with_bbox_gating.json',
        'has_bbox': True,
        'bbox_gt_file': 'test_bbox_v2.json',
        'grounding_pred': 'internal_test_grounding.json',
    },
    'prospective': {
        'img_root': 'data_external_batch2',
        'dataset': 'liver_MRI_EXTERNAL_prospective_GXMU_ready_hcc_icc_mask_64',
        'archive_name': 'prospective_gxmu',
        'qa_gt': 'test_QA_EN_multimodal_4mod_with_pred_bbox.json',
        'qa_pred': 'hcc_icc_mask_64_multi_qa_with_pred_bbox.json',
        'report_gt': 'test_report_generation_EN_multimodal_with_pred_bbox.json',
        'report_pred': 'hcc_icc_mask_64_multi_report_with_pred_bbox.json',
        'has_bbox': True,
        'grounding_pred': 'hcc_icc_mask_64_grounding.json',
    },
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
    m = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
    return list(map(int, m.groups())) if m else None


def scale_bbox(coords, W, H):
    return [
        int(coords[0] / 1000 * W),
        int(coords[1] / 1000 * H),
        int(coords[2] / 1000 * W),
        int(coords[3] / 1000 * H),
    ]


def calculate_iou(box1, box2):
    """Calculate IoU between two [x1, y1, x2, y2] boxes (already scaled)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_boxes_on_image(image_path, gt_box, pred_box, output_path, iou_score):
    """Draw GT (red) and pred (blue) boxes with IoU text using matplotlib."""
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)

        if gt_box is not None:
            x1, y1, x2, y2 = gt_box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=8, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        if pred_box is not None:
            x1, y1, x2, y2 = pred_box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=8, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

        img_width, img_height = img.size
        iou_text = f'IoU: {iou_score:.3f}'
        ax.text(img_width - 20, 20, iou_text,
                fontsize=50, fontweight='bold', color='black',
                ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor='white', edgecolor='black',
                          alpha=0.9, linewidth=2))

        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    pad_inches=0, facecolor='white', edgecolor='none')
        plt.close()
        return True
    except Exception as e:
        print(f"Error drawing boxes on {image_path}: {e}")
        plt.close()
        return False


def get_img_path(cfg, pid, mod, slc):
    """Build the full image path for a patient/modality/slice."""
    dataset = cfg['dataset']
    if dataset:
        return DATA_ROOT / cfg['img_root'] / dataset / pid / mod / "images" / f"{slc}.png"
    else:
        # internal: patient_prefix IS the folder name directly
        return DATA_ROOT / cfg['img_root'] / pid / mod / "images" / f"{slc}.png"


def extract_patient_prefix(entry, db_name):
    """Extract patient_prefix from a GT entry, handling cases where field is missing or None."""
    pp = entry.get('patient_prefix')
    if pp is not None:
        return pp
    # Fallback: extract from image path
    img = entry['image'][0]
    parts = img.split('/')
    if db_name == 'external1_gxmu_hcc_icc':
        # "liver_MRI_EXTERNAL1_HCC/589023/PRE/images/23.png" -> parts[1]
        return parts[1]
    else:
        # "D91A_0000060419/PRE/images/19.png" -> parts[0]
        return parts[0]


def parse_grounding_qid(qid, db_name):
    """Parse grounding question_id into (pid, mod, slice_str).

    Formats:
      - Most databases: "{pid}/{mod}/{slice}"
      - external1: "liver_MRI_EXTERNAL1_HCC/{pid}/{mod}/images/{slice}"
    """
    parts = qid.split('/')
    if db_name == 'external1_gxmu_hcc_icc':
        # liver_MRI_EXTERNAL1_HCC/589023/AP/images/24
        if len(parts) == 5:
            return parts[1], parts[2], parts[4]
    else:
        # pid/mod/slice
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    return None, None, None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_db_data(db_name, cfg):
    arch_name = cfg.get('archive_name', db_name)
    multi_gt_dir = ARCHIVE / "multi_gating" / arch_name / "gt"
    multi_pred_dir = ARCHIVE / "multi_gating" / arch_name / "pred"
    single_gt_dir = ARCHIVE / "single" / arch_name / "gt"
    single_pred_dir = ARCHIVE / "single" / arch_name / "pred"

    # --- QA GT (with_bbox version) ---
    qa_gt_file = cfg.get('qa_gt', 'test_QA_EN_multimodal_4mod_with_bbox.json')
    with open(multi_gt_dir / qa_gt_file) as f:
        qa_gt_all = json.load(f)

    qa_gt_by_pid = defaultdict(lambda: defaultdict(list))
    all_pids = set()
    for entry in qa_gt_all:
        pid = extract_patient_prefix(entry, db_name)
        qa_gt_by_pid[pid][entry["Question_type"]].append(entry)
        all_pids.add(pid)

    # --- QA Pred ---
    qa_pred_by_qid = {p["question_id"]: p["text"] for p in load_jsonl(multi_pred_dir / cfg['qa_pred'])}

    # --- Report GT (with_bbox version) ---
    report_gt_file = cfg.get('report_gt', 'test_report_generation_EN_multimodal_with_bbox.json')
    with open(multi_gt_dir / report_gt_file) as f:
        report_gt_all = json.load(f)

    report_gt_by_pid = {}
    for entry in report_gt_all:
        pid = extract_patient_prefix(entry, db_name)
        if pid not in report_gt_by_pid:
            report_gt_by_pid[pid] = entry

    # --- Report Pred ---
    report_pred_by_qid = {p["question_id"]: p["text"] for p in load_jsonl(multi_pred_dir / cfg['report_pred'])}

    # --- Grounding (only if has_bbox) ---
    gt_bbox = defaultdict(lambda: defaultdict(dict))
    pred_bbox = defaultdict(lambda: defaultdict(dict))

    if cfg['has_bbox']:
        bbox_gt_file = cfg.get('bbox_gt_file', 'test_bbox.json')
        with open(single_gt_dir / bbox_gt_file) as f:
            grounding_gt_all = json.load(f)

        for entry in grounding_gt_all:
            pid, mod, slc = parse_grounding_qid(entry["question_id"], db_name)
            if pid:
                coords = parse_bbox(entry["label"])
                if coords:
                    gt_bbox[pid][mod][slc] = coords

        grounding_pred_file = single_pred_dir / cfg['grounding_pred']
        if grounding_pred_file.exists() and grounding_pred_file.stat().st_size > 0:
            for entry in load_jsonl(grounding_pred_file):
                pid, mod, slc = parse_grounding_qid(entry["question_id"], db_name)
                if pid:
                    coords = parse_bbox(entry["text"])
                    if coords:
                        pred_bbox[pid][mod][slc] = coords

    return all_pids, qa_gt_by_pid, qa_pred_by_qid, report_gt_by_pid, report_pred_by_qid, gt_bbox, pred_bbox


# ---------------------------------------------------------------------------
# Per-patient export
# ---------------------------------------------------------------------------

def export_patient(pid, out_dir, cfg, db_name, qa_gt_by_pid, qa_pred_by_qid,
                   report_gt_by_pid, report_pred_by_qid, gt_bbox, pred_bbox):
    pat_dir = out_dir / f"patient_{pid}"
    img_dir = pat_dir / "images"

    # --- Collect all unique (mod, slice) from ALL QA entries ---
    all_slices = defaultdict(set)
    for qtype_entries in qa_gt_by_pid[pid].values():
        for entry in qtype_entries:
            for mod, slc in entry.get("slice_nums", {}).items():
                all_slices[mod].add(str(slc))

    # --- Copy images into per-modality subfolders ---
    for mod in MODALITIES:
        mod_img_dir = img_dir / mod
        mod_img_dir.mkdir(parents=True, exist_ok=True)
        for slc in sorted(all_slices[mod], key=lambda x: int(x)):
            src = get_img_path(cfg, pid, mod, slc)
            dst = mod_img_dir / f"{int(slc):03d}.png"
            if src.exists():
                shutil.copy(src, dst)

    # --- QA1-QA4: majority vote ---
    qa_lines = []
    for qtype in ["QA1", "QA2", "QA3", "QA4"]:
        entries = qa_gt_by_pid[pid].get(qtype, [])
        if not entries:
            continue

        gt_label = entries[0]["label"]
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

    # --- Grounding (only if db has bbox) ---
    if not cfg['has_bbox']:
        return

    for mod in pred_bbox[pid]:
        for slc in sorted(pred_bbox[pid][mod], key=lambda x: int(x)):
            pred_coords = pred_bbox[pid][mod][slc]
            gt_coords = gt_bbox[pid][mod].get(slc)

            src_img = get_img_path(cfg, pid, mod, slc)
            if not src_img.exists():
                continue

            img_base = Image.open(src_img).convert("RGB")
            W, H = img_base.size

            scaled_gt = scale_bbox(gt_coords, W, H) if gt_coords else None
            scaled_pred = scale_bbox(pred_coords, W, H) if pred_coords else None

            iou = calculate_iou(scaled_gt, scaled_pred) if scaled_gt and scaled_pred else 0.0

            slc_int = int(slc)

            # GT only (red)
            grd_gt_dir = pat_dir / "grounding_gt" / mod
            grd_gt_dir.mkdir(parents=True, exist_ok=True)
            img0 = img_base.copy()
            draw0 = ImageDraw.Draw(img0)
            if scaled_gt:
                draw0.rectangle(scaled_gt, outline="red", width=8)
            img0.save(grd_gt_dir / f"{slc_int:03d}.png")

            # Pred only (blue)
            grd_pred_dir = pat_dir / "grounding_pred" / mod
            grd_pred_dir.mkdir(parents=True, exist_ok=True)
            img1 = img_base.copy()
            draw1 = ImageDraw.Draw(img1)
            if scaled_pred:
                draw1.rectangle(scaled_pred, outline="blue", width=8)
            img1.save(grd_pred_dir / f"{slc_int:03d}.png")

            # Both: GT (red) + pred (blue) + IoU text (matplotlib)
            grd_both_dir = pat_dir / "grounding_both" / mod
            grd_both_dir.mkdir(parents=True, exist_ok=True)
            draw_boxes_on_image(
                src_img, scaled_gt, scaled_pred,
                grd_both_dir / f"{slc_int:03d}.png", iou)


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

LABEL_NUM = {
    'D. No focal lesion': 0,
    'C. Primary hepatocellular carcinoma (HCC)': 1,
    'A. Intrahepatic cholangiocarcinoma (ICC)': 2,
    'B. Cavernous hemangioma or hepatic cyst (Benign)': 3,
}


def save_qa4_csv(out_dir, all_pids, qa_gt_by_pid, qa_pred_by_qid):
    rows = []
    for pid in sorted(all_pids):
        entries = qa_gt_by_pid[pid].get("QA4", [])
        if not entries:
            continue
        gt_label = entries[0]["label"]
        preds = []
        for entry in entries:
            pred_text = qa_pred_by_qid.get(entry["question_id"])
            if pred_text:
                preds.append(pred_text)
        pred_label = majority_vote(preds) if preds else ""
        rows.append({
            'patient_id': pid,
            'gt': LABEL_NUM.get(gt_label, -1),
            'pred': LABEL_NUM.get(pred_label, -1),
        })

    csv_path = out_dir / "qa4_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'gt', 'pred'])
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(1 for r in rows if r['gt'] == r['pred'])
    total = len(rows)
    acc = correct / total if total else 0
    print(f"    QA4 CSV: {total} patients, acc={correct}/{total}={acc:.4f} -> {csv_path}")


def save_gt_bbox_csv(out_dir, gt_bbox):
    rows = []
    for pid in sorted(gt_bbox):
        for mod in MODALITIES:
            for slc in sorted(gt_bbox[pid][mod], key=lambda x: int(x)):
                coords = gt_bbox[pid][mod][slc]
                rows.append({
                    'patient_id': pid,
                    'modality': mod,
                    'slice': int(slc),
                    'x1': coords[0], 'y1': coords[1],
                    'x2': coords[2], 'y2': coords[3],
                })

    if not rows:
        return

    csv_path = out_dir / "gt_bbox.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'modality', 'slice',
                                                'x1', 'y1', 'x2', 'y2'])
        writer.writeheader()
        writer.writerows(rows)

    n_patients = len({r['patient_id'] for r in rows})
    print(f"    GT bbox CSV: {len(rows)} entries from {n_patients} patients -> {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export all database results")
    parser.add_argument("--output", default="groundliver_results",
                        help="Output root directory (default: groundliver_results)")
    parser.add_argument("--db", nargs="*", default=None,
                        help="Specific database(s) to export (default: all)")
    args = parser.parse_args()

    root_dir = Path(args.output)
    db_names = args.db if args.db else list(DB_CONFIG.keys())

    for db_name in db_names:
        if db_name not in DB_CONFIG:
            print(f"Unknown database: {db_name}, skipping")
            continue

        cfg = DB_CONFIG[db_name]
        print(f"\n{'='*60}")
        print(f"Processing: {db_name}")
        print(f"{'='*60}")

        print("  Loading data...")
        all_pids, qa_gt_by_pid, qa_pred_by_qid, report_gt_by_pid, report_pred_by_qid, gt_bbox, pred_bbox = load_db_data(db_name, cfg)

        db_dir = root_dir / db_name
        db_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Exporting {len(all_pids)} patients to {db_dir}/")
        for pid in sorted(all_pids):
            export_patient(
                pid, db_dir, cfg, db_name,
                qa_gt_by_pid, qa_pred_by_qid,
                report_gt_by_pid, report_pred_by_qid,
                gt_bbox, pred_bbox,
            )

        # Save CSVs
        save_qa4_csv(db_dir, all_pids, qa_gt_by_pid, qa_pred_by_qid)
        if cfg['has_bbox']:
            save_gt_bbox_csv(db_dir, gt_bbox)

        print(f"  Done: {db_name} ({len(all_pids)} patients, bbox={'YES' if cfg['has_bbox'] else 'NO'})")

    print(f"\nAll done. Results in {root_dir}/")


if __name__ == "__main__":
    main()
