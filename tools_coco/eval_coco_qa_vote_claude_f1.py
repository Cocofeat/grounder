import json
import jsonlines
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re


def load_predictions(pred_file: str) -> List[Dict]:
    """Load predictions from JSONL file"""
    predictions = []
    with jsonlines.open(pred_file, 'r') as reader:
        for obj in reader:
            predictions.append(obj)
    return predictions


def load_ground_truth(gt_file: str) -> List[Dict]:
    """Load ground truth from JSON file"""
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth


def parse_image_path(image_path: str) -> Tuple[str, str, str]:
    """Parse image path to extract patient_id, modality, and instance"""
    # Example: D91A_0000060419/AP/images/19.png
    parts = image_path.split('/')
    patient_id = parts[0]
    modality = parts[1]
    instance = parts[3]  # includes .png extension
    return patient_id, modality, instance


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate accuracy between predictions and ground truths"""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def extract_option_letter(text: str) -> str:
    """Extract leading option letter (A/B/C/D) from 'A', 'A.', 'A:', 'A )', or 'A xxx'.

    If not found at the start, falls back to the first A-D letter appearing in text.
    Returns empty string if none found.
    """
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace('：', ':')
    m = re.match(r"^\s*([A-Da-d])([\.:、\)\s]|$)", s)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"([A-Da-d])", s)
    return m2.group(1).upper() if m2 else ""


def compute_f1_scores(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Compute per-class, macro, micro, and weighted F1 for single-label classification.

    Returns a dict with keys: per_class, macro, micro, weighted, support, labels
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have same length"
    n = len(ground_truths)
    if n == 0:
        return {
            'per_class': {},
            'macro': 0.0,
            'micro': 0.0,
            'weighted': 0.0,
            'support': {},
            'labels': []
        }

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

    per_class = {}
    macro_f1_sum = 0.0
    macro_count = 0
    weighted_f1_sum = 0.0
    total_support = sum(support.values())

    for l in labels:
        p_denom = tp[l] + fp[l]
        r_denom = tp[l] + fn[l]
        precision = tp[l] / p_denom if p_denom > 0 else 0.0
        recall = tp[l] / r_denom if r_denom > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class[l] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support[l]
        }
        if support[l] > 0:
            macro_f1_sum += f1
            macro_count += 1
            weighted_f1_sum += f1 * support[l]

    macro = (macro_f1_sum / macro_count) if macro_count > 0 else 0.0

    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_prec_denom = micro_tp + micro_fp
    micro_rec_denom = micro_tp + micro_fn
    micro_precision = micro_tp / micro_prec_denom if micro_prec_denom > 0 else 0.0
    micro_recall = micro_tp / micro_rec_denom if micro_rec_denom > 0 else 0.0
    micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    weighted = (weighted_f1_sum / total_support) if total_support > 0 else 0.0

    return {
        'per_class': per_class,
        'macro': macro,
        'micro': micro,
        'weighted': weighted,
        'support': support,
        'labels': labels
    }


def majority_vote(votes: List[str]) -> str:
    """Return majority vote from a list of predictions"""
    if not votes:
        return ""

    vote_counts = Counter(votes)
    # Return the most common vote
    return vote_counts.most_common(1)[0][0]


def evaluate_medical_qa_with_f1(pred_file: str, gt_file: str):
    """Main evaluation function with F1 reporting"""

    # Load data
    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")

    # Verify matching order
    if len(predictions) != len(ground_truth):
        print(f"Warning: Mismatch in number of entries - pred: {len(predictions)}, gt: {len(ground_truth)}")

    # Organize data by different dimensions
    data_by_question_type = defaultdict(lambda: {'pred': [], 'gt': []})
    data_by_modality = defaultdict(lambda: {'pred': [], 'gt': []})
    data_by_patient_modality = defaultdict(lambda: defaultdict(lambda: {'pred': [], 'gt': [], 'question_types': []}))
    data_by_patient_question_type = defaultdict(lambda: defaultdict(lambda: {'pred': [], 'gt': [], 'modalities': []}))

    # Process each prediction-ground truth pair
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if pred['question_id'] != gt['question_id']:
            print(f"Warning: Question ID mismatch at index {i}: {pred['question_id']} vs {gt['question_id']}")
            continue

        pred_text = pred['text']
        gt_text = gt['label']
        question_type = gt['Question_type']
        modality = gt['modality']

        # Parse patient info
        patient_id, modality_parsed, instance = parse_image_path(gt['image'])

        # Store by question type
        data_by_question_type[question_type]['pred'].append(pred_text)
        data_by_question_type[question_type]['gt'].append(gt_text)

        # Store by modality
        data_by_modality[modality]['pred'].append(pred_text)
        data_by_modality[modality]['gt'].append(gt_text)

        # Store by patient+modality for voting
        key_pm = f"{patient_id}_{modality}"
        data_by_patient_modality[key_pm][question_type]['pred'].append(pred_text)
        data_by_patient_modality[key_pm][question_type]['gt'].append(gt_text)
        data_by_patient_modality[key_pm][question_type]['question_types'].append(question_type)

        # Store by patient+question_type for cross-modality voting
        key_pq = f"{patient_id}_{question_type}"
        data_by_patient_question_type[key_pq][modality]['pred'].append(pred_text)
        data_by_patient_question_type[key_pq][modality]['gt'].append(gt_text)
        data_by_patient_question_type[key_pq][modality]['modalities'].append(modality)

    # Overall performance (no voting)
    print("\n=== Overall Performance (No Voting) ===")
    all_preds = []
    all_gts = []
    for qtype, data in data_by_question_type.items():
        all_preds.extend(data['pred'])
        all_gts.extend(data['gt'])
    if all_preds:
        acc_overall = calculate_accuracy(all_preds, all_gts)
        f1_overall = compute_f1_scores(all_preds, all_gts)
        print(f"Accuracy: {acc_overall:.4f}")
        print(f"F1 - macro: {f1_overall['macro']:.4f}, micro: {f1_overall['micro']:.4f}, weighted: {f1_overall['weighted']:.4f}")

    # 1. Evaluate by Question Type
    print("\n=== Performance by Question Type ===")
    question_type_results = {}
    for question_type, data in data_by_question_type.items():
        accuracy = calculate_accuracy(data['pred'], data['gt'])
        f1_stats = compute_f1_scores(data['pred'], data['gt'])
        question_type_results[question_type] = {
            'accuracy': accuracy,
            'f1': f1_stats
        }
        print(
            f"{question_type}: Acc {accuracy:.4f} | F1(macro/micro/wt) "
            f"{f1_stats['macro']:.4f}/{f1_stats['micro']:.4f}/{f1_stats['weighted']:.4f} "
            f"({len(data['pred'])} samples)"
        )

    # 2. Evaluate by Modality
    print("\n=== Performance by Modality ===")
    modality_results = {}
    for modality, data in data_by_modality.items():
        accuracy = calculate_accuracy(data['pred'], data['gt'])
        f1_stats = compute_f1_scores(data['pred'], data['gt'])
        modality_results[modality] = {
            'accuracy': accuracy,
            'f1': f1_stats
        }
        print(
            f"{modality}: Acc {accuracy:.4f} | F1(macro/micro/wt) "
            f"{f1_stats['macro']:.4f}/{f1_stats['micro']:.4f}/{f1_stats['weighted']:.4f} "
            f"({len(data['pred'])} samples)"
        )

    # 3. Evaluate with voting within modality (across instances)
    print("\n=== Performance with Voting within Modality ===")
    modality_voting_results = {}
    for modality in modality_results.keys():
        modality_voting_results[modality] = {}

        # Group by patient+modality+question_type and vote across instances
        patient_modality_votes = defaultdict(lambda: defaultdict(lambda: {'pred_votes': [], 'gt_vote': None}))

        for patient_modality_key, question_data in data_by_patient_modality.items():
            if not patient_modality_key.endswith(f"_{modality}"):
                continue

            patient_id = patient_modality_key.rsplit('_', 1)[0]

            for question_type, qa_data in question_data.items():
                if qa_data['pred']:  # If there are predictions for this question type
                    patient_modality_votes[patient_id][question_type]['pred_votes'] = qa_data['pred']
                    patient_modality_votes[patient_id][question_type]['gt_vote'] = qa_data['gt'][0]  # Should be same for all instances

        # Calculate voting metrics for each question type within this modality
        for question_type in question_type_results.keys():
            voted_preds = []
            voted_gts = []

            for patient_id, question_data in patient_modality_votes.items():
                if question_type in question_data and question_data[question_type]['pred_votes']:
                    voted_pred = majority_vote(question_data[question_type]['pred_votes'])
                    voted_gt = question_data[question_type]['gt_vote']

                    voted_preds.append(voted_pred)
                    voted_gts.append(voted_gt)

            if voted_preds:
                voting_accuracy = calculate_accuracy(voted_preds, voted_gts)
                voting_f1 = compute_f1_scores(voted_preds, voted_gts)
                modality_voting_results[modality][question_type] = {
                    'accuracy': voting_accuracy,
                    'f1': voting_f1
                }
                print(
                    f"{modality} - {question_type}: Acc {voting_accuracy:.4f} | F1(macro/micro/wt) "
                    f"{voting_f1['macro']:.4f}/{voting_f1['micro']:.4f}/{voting_f1['weighted']:.4f} "
                    f"({len(voted_preds)} patients)"
                )

    # 4. Evaluate with voting across modalities
    print("\n=== Performance with Voting across Modalities ===")
    cross_modality_results = {}

    # Group by patient+question_type and vote across modalities
    patient_question_votes = defaultdict(lambda: {'pred_votes': [], 'gt_vote': None})

    for patient_question_key, modality_data in data_by_patient_question_type.items():
        # patient_id, question_type = patient_question_key.rsplit('_', 1)

        # Collect votes from all modalities for this patient+question
        all_votes = []
        gt_vote = None

        for modality, qa_data in modality_data.items():
            if qa_data['pred']:
                # Vote within modality first (across instances)
                modality_vote = majority_vote(qa_data['pred'])
                all_votes.append(modality_vote)

                if gt_vote is None:
                    gt_vote = qa_data['gt'][0]  # Should be same across modalities

        if all_votes:
            patient_question_votes[patient_question_key] = {
                'pred_votes': all_votes,
                'gt_vote': gt_vote
            }

    # Calculate cross-modality voting metrics for each question type
    for question_type in question_type_results.keys():
        cross_modal_preds = []
        cross_modal_gts = []

        for patient_question_key, vote_data in patient_question_votes.items():
            if patient_question_key.endswith(f"_{question_type}"):
                voted_pred = majority_vote(vote_data['pred_votes'])
                voted_gt = vote_data['gt_vote']

                cross_modal_preds.append(voted_pred)
                cross_modal_gts.append(voted_gt)

        if cross_modal_preds:
            cross_accuracy = calculate_accuracy(cross_modal_preds, cross_modal_gts)
            cross_f1 = compute_f1_scores(cross_modal_preds, cross_modal_gts)
            cross_modality_results[question_type] = {
                'accuracy': cross_accuracy,
                'f1': cross_f1
            }
            print(
                f"{question_type}: Acc {cross_accuracy:.4f} | F1(macro/micro/wt) "
                f"{cross_f1['macro']:.4f}/{cross_f1['micro']:.4f}/{cross_f1['weighted']:.4f} "
                f"({len(cross_modal_preds)} patients)"
            )

    # 5. Answer distribution counts (A/B/C[/D])
    print("\n=== Answer Distribution (Counts) ===")

    def tally_counts(items: List[str]) -> Dict[str, int]:
        cnt = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for x in items:
            l = extract_option_letter(x)
            if l in cnt:
                cnt[l] += 1
        return cnt

    # Ground Truth distributions
    print("-- Ground Truth --")
    gt_all = []
    for qtype, data in data_by_question_type.items():
        gt_all.extend(data['gt'])
    gt_all_cnt = tally_counts(gt_all)
    total_gt = sum(gt_all_cnt.values())
    print(f"Overall: A {gt_all_cnt['A']}, B {gt_all_cnt['B']}, C {gt_all_cnt['C']}, D {gt_all_cnt['D']} (N={total_gt})")
    for qtype, data in data_by_question_type.items():
        c = tally_counts(data['gt'])
        n = sum(c.values())
        print(f"{qtype}: A {c['A']}, B {c['B']}, C {c['C']}, D {c['D']} (N={n})")

    # Prediction distributions (no voting)
    print("\n-- Predictions (No Voting) --")
    pred_all = []
    for qtype, data in data_by_question_type.items():
        pred_all.extend(data['pred'])
    pred_all_cnt = tally_counts(pred_all)
    total_pred = sum(pred_all_cnt.values())
    print(f"Overall: A {pred_all_cnt['A']}, B {pred_all_cnt['B']}, C {pred_all_cnt['C']}, D {pred_all_cnt['D']} (N={total_pred})")
    for qtype, data in data_by_question_type.items():
        c = tally_counts(data['pred'])
        n = sum(c.values())
        print(f"{qtype}: A {c['A']}, B {c['B']}, C {c['C']}, D {c['D']} (N={n})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate medical QA with majority voting and F1 scores")
    # parser.add_argument('--pred', default="playground/coco/total_full_qa.json", help='Path to predictions JSONL file')
    # parser.add_argument('--gt', default="/mnt/data/by/data/coco_new/labels/test_QA_EN.json", help='Path to ground truth JSON file')
    parser.add_argument('--pred', default="playground/coco/total_full_qa_bbox.json", help='Path to predictions JSONL file')
    parser.add_argument('--gt', default="/mnt/data/by/data/coco_new/labels/test_QA_EN_with_bbox.json.bak", help='Path to ground truth JSON file')
    args = parser.parse_args()
    # external_lora_qa.json
    evaluate_medical_qa_with_f1(args.pred, args.gt)
