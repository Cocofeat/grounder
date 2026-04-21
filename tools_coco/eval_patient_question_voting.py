"""
Evaluate predictions with voting: for the same patient + same question type,
aggregate all predictions (across different images) and do majority voting.

Then compute Accuracy and F1 Macro for each question type.
"""

import argparse
import json
import jsonlines
from collections import defaultdict, Counter
import re


def load_predictions(pred_file: str):
    """Load predictions from JSONL file."""
    predictions = []
    with jsonlines.open(pred_file, 'r') as reader:
        for obj in reader:
            predictions.append(obj)
    return predictions


def load_ground_truth(gt_file: str):
    """Load ground truth from JSON file."""
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth


def extract_patient_id(question_id: str) -> str:
    """Extract patient ID from question_id like '589023/AP/images/24'."""
    return question_id.split('/')[0]


def strip_bbox_prefix(prompt: str) -> str:
    """Remove bbox prefix like 'Lesion is located at <box>[[...]]</box>. ' from prompt."""
    pattern = r'^Lesion is located at <box>\[\[.*?\]\]</box>\.\s*'
    return re.sub(pattern, '', prompt)


def extract_option_letter(text: str) -> str:
    """Extract leading option letter (A/B/C/D/E) from answer text."""
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace('：', ':')
    m = re.match(r"^\s*([A-Ea-e])([\.:、\)\s]|$)", s)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"([A-Ea-e])", s)
    return m2.group(1).upper() if m2 else ""


def majority_vote(votes: list) -> str:
    """Return majority vote from a list of predictions."""
    if not votes:
        return ""
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


def compute_f1_macro(predictions: list, ground_truths: list) -> float:
    """Compute macro F1 score."""
    if len(predictions) != len(ground_truths) or len(predictions) == 0:
        return 0.0

    labels = sorted(list(set(ground_truths) | set(predictions)))
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    support = {l: 0 for l in labels}

    for pred, gt in zip(predictions, ground_truths):
        support[gt] = support.get(gt, 0) + 1
        if pred == gt:
            tp[gt] = tp.get(gt, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[gt] = fn.get(gt, 0) + 1

    macro_f1_sum = 0.0
    macro_count = 0

    for l in labels:
        p_denom = tp[l] + fp[l]
        r_denom = tp[l] + fn[l]
        precision = tp[l] / p_denom if p_denom > 0 else 0.0
        recall = tp[l] / r_denom if r_denom > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        if support[l] > 0:
            macro_f1_sum += f1
            macro_count += 1

    return (macro_f1_sum / macro_count) if macro_count > 0 else 0.0


def calculate_accuracy(predictions: list, ground_truths: list) -> float:
    """Calculate accuracy."""
    if len(predictions) != len(ground_truths) or len(predictions) == 0:
        return 0.0
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(predictions)


def evaluate_with_patient_question_voting(pred_file: str, gt_file: str):
    """
    For the same patient + same question type, aggregate all predictions and do voting.
    Then compute Accuracy and F1 Macro per question type.
    """
    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground-truth entries")

    # Build GT lookup: (question_id, Question_type) -> gt_entry
    gt_lookup = {}
    for gt in ground_truth:
        qid = gt['question_id']
        qtype = gt.get('Question_type', gt.get('type', 'UNKNOWN'))
        key = (qid, qtype)
        gt_lookup[key] = gt

    # Aggregate predictions by (patient_id, question_type)
    # Structure: {(patient_id, question_type): {'preds': [], 'gt_label': label}}
    patient_question_data = defaultdict(lambda: {'preds': [], 'gt_label': None})

    matched = 0
    unmatched = 0

    for pred in predictions:
        pred_qid = pred['question_id']
        pred_text = pred.get('text', '')

        # We need to find the matching GT entry to get question_type
        # Since prediction file doesn't have Question_type, we need to match by prompt/text
        # Or iterate through GT to find matching question_id and prompt

        # Find matching GT entries with this question_id
        matching_gts = [(k, v) for k, v in gt_lookup.items() if k[0] == pred_qid]

        if not matching_gts:
            unmatched += 1
            continue

        # Match by prompt/question text (strip bbox prefix if present)
        pred_prompt = pred.get('prompt', '')
        pred_prompt_clean = strip_bbox_prefix(pred_prompt)
        found_match = None
        for (qid, qtype), gt_entry in matching_gts:
            gt_question = gt_entry.get('text', '')
            if pred_prompt_clean == gt_question or pred_prompt == gt_question:
                found_match = (qid, qtype, gt_entry)
                break

        if found_match is None:
            unmatched += 1
            continue

        qid, qtype, gt_entry = found_match
        patient_id = extract_patient_id(qid)
        gt_label = extract_option_letter(gt_entry['label'])
        pred_label = extract_option_letter(pred_text)

        key = (patient_id, qtype)
        patient_question_data[key]['preds'].append(pred_label)
        patient_question_data[key]['gt_label'] = gt_label

        matched += 1

    print(f"Matched: {matched}, Unmatched: {unmatched}")

    # Now do voting for each (patient, question_type)
    # Group by question_type for final evaluation
    question_type_results = defaultdict(lambda: {'voted_preds': [], 'gts': []})

    for (patient_id, qtype), data in patient_question_data.items():
        if not data['preds'] or data['gt_label'] is None:
            continue

        voted_pred = majority_vote(data['preds'])
        gt_label = data['gt_label']

        question_type_results[qtype]['voted_preds'].append(voted_pred)
        question_type_results[qtype]['gts'].append(gt_label)

    # Calculate metrics per question type
    print("\n" + "=" * 60)
    print("Results: Voting per (Patient, Question Type)")
    print("=" * 60)

    all_preds = []
    all_gts = []

    for qtype in sorted(question_type_results.keys()):
        data = question_type_results[qtype]
        preds = data['voted_preds']
        gts = data['gts']

        acc = calculate_accuracy(preds, gts)
        f1_macro = compute_f1_macro(preds, gts)

        print(f"{qtype}: Accuracy={acc:.4f}, F1_Macro={f1_macro:.4f} (N={len(preds)} patients)")

        all_preds.extend(preds)
        all_gts.extend(gts)

    # Overall
    if all_preds:
        overall_acc = calculate_accuracy(all_preds, all_gts)
        overall_f1 = compute_f1_macro(all_preds, all_gts)
        print("-" * 60)
        print(f"Overall: Accuracy={overall_acc:.4f}, F1_Macro={overall_f1:.4f} (N={len(all_preds)} total)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate with voting per (patient, question_type)."
    )
    parser.add_argument('--pred', required=True, help='Path to predictions JSONL file')
    parser.add_argument('--gt', required=True, help='Path to ground truth JSON file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_with_patient_question_voting(args.pred, args.gt)
