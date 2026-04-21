"""
Evaluation script for visual grounding (bounding box prediction).

Supports both single-modal and multi-modal grounding evaluation.
- Single-modal: one image -> one bbox
- Multi-modal: 4 images -> bbox (based on one modality, typically AP)

Metrics: mIoU, IoU@0.1, IoU@0.3, IoU@0.5
"""

import json
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import argparse


def parse_bounding_box_single(text: str) -> Optional[List[int]]:
    """
    Extract single bounding box from text containing <box>[[x1, y1, x2, y2]]</box> format.
    """
    if text is None:
        return None

    pattern = r'<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>'
    match = re.search(pattern, text)

    if match:
        return [int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))]
    return None


def parse_bounding_box_multi(text: str) -> Optional[List[List[int]]]:
    """
    Extract multiple bounding boxes from text containing <box>[[x1,y1,x2,y2], [x1,y1,x2,y2], ...]</box>.
    Returns list of 4 bboxes (one per modality).
    """
    if text is None:
        return None

    # Pattern for multiple boxes
    pattern = r'<box>\[((?:\[[\d,\s]+\],?\s*)+)\]</box>'
    match = re.search(pattern, text)

    if match:
        boxes_str = match.group(1)
        # Find all individual boxes
        box_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        boxes = re.findall(box_pattern, boxes_str)

        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]
    return None


def calculate_iou(box1: List[int], box2: List[int]) -> Optional[float]:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Returns None if GT box (box2) is [0,0,0,0] to exclude from evaluation.
    """
    if box1 is None or box2 is None:
        return None

    # Skip samples where GT is [0, 0, 0, 0] (no lesion)
    if sum(box2) == 0:
        return None

    # If prediction is [0, 0, 0, 0] but GT is not, IoU = 0
    if sum(box1) == 0:
        return 0.0

    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def load_predictions(jsonl_path: str) -> Dict[str, str]:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                predictions[data['question_id']] = data['text']
    return predictions


def load_ground_truth(json_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load ground truth from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    ground_truth = {}
    modalities = {}
    patient_labels = {}

    for item in gt_data:
        qid = item['question_id']
        ground_truth[qid] = item['label']
        # For single-modal: modality field; for multi-modal: modalities list
        if 'modality' in item:
            modalities[qid] = item['modality']
        elif 'modalities' in item:
            modalities[qid] = item['modalities']
        patient_labels[qid] = item.get('patient_label', 'unknown')

    return ground_truth, modalities, patient_labels


def evaluate_single_grounding(pred_file: str, gt_file: str) -> Dict:
    """Evaluate single-modal grounding."""
    predictions = load_predictions(pred_file)
    ground_truth, modalities, patient_labels = load_ground_truth(gt_file)

    common_ids = set(predictions.keys()) & set(ground_truth.keys())

    if not common_ids:
        print("No matching question IDs found!")
        return {}

    print(f"Evaluating {len(common_ids)} samples...")

    # Overall tracking
    all_ious = []
    iou_thresholds = [0.1, 0.3, 0.5]
    threshold_counts = {t: 0 for t in iou_thresholds}

    # By modality
    modality_stats = defaultdict(lambda: {'ious': [], 'counts': {t: 0 for t in iou_thresholds}})

    # By patient label (HCC, ICC, etc.)
    label_stats = defaultdict(lambda: {'ious': [], 'counts': {t: 0 for t in iou_thresholds}})

    invalid_count = 0

    skipped_zero_gt = 0

    for qid in common_ids:
        pred_box = parse_bounding_box_single(predictions[qid])
        gt_box = parse_bounding_box_single(ground_truth[qid])

        if pred_box is None or gt_box is None:
            invalid_count += 1
            continue

        iou = calculate_iou(pred_box, gt_box)

        # Skip if GT is [0,0,0,0] (no lesion)
        if iou is None:
            skipped_zero_gt += 1
            continue

        all_ious.append(iou)

        mod = modalities.get(qid, 'unknown')
        plabel = patient_labels.get(qid, 'unknown')

        modality_stats[mod]['ious'].append(iou)
        label_stats[plabel]['ious'].append(iou)

        for t in iou_thresholds:
            if iou >= t:
                threshold_counts[t] += 1
                modality_stats[mod]['counts'][t] += 1
                label_stats[plabel]['counts'][t] += 1

    valid_count = len(all_ious)

    # Compute results
    results = {
        'overall': {
            'mIoU': np.mean(all_ious) if all_ious else 0.0,
            'IoU@0.1': threshold_counts[0.1] / valid_count if valid_count else 0.0,
            'IoU@0.3': threshold_counts[0.3] / valid_count if valid_count else 0.0,
            'IoU@0.5': threshold_counts[0.5] / valid_count if valid_count else 0.0,
            'total': len(common_ids),
            'valid': valid_count,
            'invalid': invalid_count,
            'skipped_zero_gt': skipped_zero_gt
        },
        'by_modality': {},
        'by_label': {}
    }

    for mod, stats in modality_stats.items():
        n = len(stats['ious'])
        if n > 0:
            results['by_modality'][mod] = {
                'mIoU': np.mean(stats['ious']),
                'IoU@0.1': stats['counts'][0.1] / n,
                'IoU@0.3': stats['counts'][0.3] / n,
                'IoU@0.5': stats['counts'][0.5] / n,
                'count': n
            }

    for label, stats in label_stats.items():
        n = len(stats['ious'])
        if n > 0:
            results['by_label'][label] = {
                'mIoU': np.mean(stats['ious']),
                'IoU@0.1': stats['counts'][0.1] / n,
                'IoU@0.3': stats['counts'][0.3] / n,
                'IoU@0.5': stats['counts'][0.5] / n,
                'count': n
            }

    return results


def evaluate_multi_grounding(pred_file: str, gt_file: str) -> Dict:
    """Evaluate multi-modal grounding.

    For multi-modal, both prediction and GT have 4 bboxes (one per modality).
    We compare each modality's bbox separately and also compute overall metrics.
    Order: PRE, AP, PVP, T2WI (index 0, 1, 2, 3)
    """
    predictions = load_predictions(pred_file)
    ground_truth, modalities, patient_labels = load_ground_truth(gt_file)

    common_ids = set(predictions.keys()) & set(ground_truth.keys())

    if not common_ids:
        print("No matching question IDs found!")
        return {}

    print(f"Evaluating {len(common_ids)} samples...")

    iou_thresholds = [0.1, 0.3, 0.5]

    # By modality (for multi-modal evaluation)
    modality_names = ['PRE', 'AP', 'PVP', 'T2WI']
    modality_stats = {m: {'ious': [], 'counts': {t: 0 for t in iou_thresholds}} for m in modality_names}

    # By patient label (store avg IoU per sample)
    label_stats = defaultdict(lambda: {'ious': [], 'counts': {t: 0 for t in iou_thresholds}})

    invalid_count = 0
    all_avg_ious = []
    threshold_counts = {t: 0 for t in iou_thresholds}

    for qid in common_ids:
        pred_text = predictions[qid]
        gt_text = ground_truth[qid]

        # Parse multi-modal prediction (4 bboxes)
        pred_boxes = parse_bounding_box_multi(pred_text)
        # GT can be single bbox or 4 bboxes
        gt_boxes = parse_bounding_box_multi(gt_text)
        if gt_boxes is None:
            gt_single = parse_bounding_box_single(gt_text)
            if gt_single:
                gt_boxes = [gt_single]

        if pred_boxes is None or gt_boxes is None:
            invalid_count += 1
            continue

        # Get modality order from GT
        mods = modalities.get(qid, ['PRE', 'AP', 'PVP', 'T2WI'])
        # Ensure mods is a list
        if not isinstance(mods, list):
            mods = ['PRE', 'AP', 'PVP', 'T2WI']

        # Calculate IoU for each modality
        sample_ious = []
        for i, mod in enumerate(mods):
            if mod not in modality_stats:
                modality_stats[mod] = {'ious': [], 'counts': {t: 0 for t in iou_thresholds}}

            if i < len(pred_boxes) and i < len(gt_boxes):
                iou = calculate_iou(pred_boxes[i], gt_boxes[i])
                # Skip if GT is [0,0,0,0]
                if iou is None:
                    continue
                sample_ious.append(iou)
                modality_stats[mod]['ious'].append(iou)
                for t in iou_thresholds:
                    if iou >= t:
                        modality_stats[mod]['counts'][t] += 1
            elif i < len(pred_boxes) and len(gt_boxes) == 1:
                if mod == 'AP':
                    iou = calculate_iou(pred_boxes[i], gt_boxes[0])
                    # Skip if GT is [0,0,0,0]
                    if iou is None:
                        continue
                    sample_ious.append(iou)
                    modality_stats[mod]['ious'].append(iou)
                    for t in iou_thresholds:
                        if iou >= t:
                            modality_stats[mod]['counts'][t] += 1

        if not sample_ious:
            invalid_count += 1
            continue

        # Average IoU across modalities for this sample
        avg_iou = sum(sample_ious) / len(sample_ious)
        all_avg_ious.append(avg_iou)

        for t in iou_thresholds:
            if avg_iou >= t:
                threshold_counts[t] += 1

        # Track by patient label
        plabel = patient_labels.get(qid, 'unknown')
        label_stats[plabel]['ious'].append(avg_iou)
        for t in iou_thresholds:
            if avg_iou >= t:
                label_stats[plabel]['counts'][t] += 1

    valid_count = len(all_avg_ious)

    results = {
        'overall': {
            'mIoU': np.mean(all_avg_ious) if all_avg_ious else 0.0,
            'IoU@0.1': threshold_counts[0.1] / valid_count if valid_count else 0.0,
            'IoU@0.3': threshold_counts[0.3] / valid_count if valid_count else 0.0,
            'IoU@0.5': threshold_counts[0.5] / valid_count if valid_count else 0.0,
            'total': len(common_ids),
            'valid': valid_count,
            'invalid': invalid_count
        },
        'by_modality': {},
        'by_label': {}
    }

    for mod, stats in modality_stats.items():
        n = len(stats['ious'])
        if n > 0:
            results['by_modality'][mod] = {
                'mIoU': np.mean(stats['ious']),
                'IoU@0.1': stats['counts'][0.1] / n,
                'IoU@0.3': stats['counts'][0.3] / n,
                'IoU@0.5': stats['counts'][0.5] / n,
                'count': n
            }

    for label, stats in label_stats.items():
        n = len(stats['ious'])
        if n > 0:
            results['by_label'][label] = {
                'mIoU': np.mean(stats['ious']),
                'IoU@0.1': stats['counts'][0.1] / n,
                'IoU@0.3': stats['counts'][0.3] / n,
                'IoU@0.5': stats['counts'][0.5] / n,
                'count': n
            }

    return results


def print_results(results: Dict, title: str):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    overall = results['overall']
    print(f"\nOverall Performance:")
    skipped_info = f", Skipped (GT=0): {overall['skipped_zero_gt']}" if 'skipped_zero_gt' in overall else ""
    print(f"  Total samples: {overall['total']}, Valid: {overall['valid']}, Invalid: {overall['invalid']}{skipped_info}")
    print(f"  mIoU: {overall['mIoU']:.4f}")
    print(f"  IoU@0.1: {overall['IoU@0.1']:.4f} ({overall['IoU@0.1']*100:.1f}%)")
    print(f"  IoU@0.3: {overall['IoU@0.3']:.4f} ({overall['IoU@0.3']*100:.1f}%)")
    print(f"  IoU@0.5: {overall['IoU@0.5']:.4f} ({overall['IoU@0.5']*100:.1f}%)")

    if 'by_modality' in results and results['by_modality']:
        print(f"\nBy Modality:")
        print(f"  {'Modality':<10} {'Count':<8} {'mIoU':<8} {'IoU@0.1':<8} {'IoU@0.3':<8} {'IoU@0.5':<8}")
        print("  " + "-" * 58)
        for mod in sorted(results['by_modality'].keys()):
            r = results['by_modality'][mod]
            print(f"  {mod:<10} {r['count']:<8} {r['mIoU']:<8.4f} {r['IoU@0.1']:<8.4f} {r['IoU@0.3']:<8.4f} {r['IoU@0.5']:<8.4f}")

    if 'by_label' in results and results['by_label']:
        print(f"\nBy Patient Label:")
        print(f"  {'Label':<15} {'Count':<8} {'mIoU':<8} {'IoU@0.1':<8} {'IoU@0.3':<8} {'IoU@0.5':<8}")
        print("  " + "-" * 63)
        for label in sorted(results['by_label'].keys()):
            r = results['by_label'][label]
            print(f"  {label:<15} {r['count']:<8} {r['mIoU']:<8.4f} {r['IoU@0.1']:<8.4f} {r['IoU@0.3']:<8.4f} {r['IoU@0.5']:<8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate visual grounding")
    parser.add_argument('--pred', required=True, help='Path to predictions JSONL file')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON file')
    parser.add_argument('--mode', choices=['single', 'multi'], default='single',
                        help='Evaluation mode: single or multi modal')
    args = parser.parse_args()

    if args.mode == 'single':
        results = evaluate_single_grounding(args.pred, args.gt)
        print_results(results, "Single-Modal Visual Grounding Evaluation")
    else:
        results = evaluate_multi_grounding(args.pred, args.gt)
        print_results(results, "Multi-Modal Visual Grounding Evaluation")

    return results


if __name__ == "__main__":
    main()
