"""
Evaluation script for single-image report generation.

This script evaluates predictions with patient-level aggregation:
- For each patient, select the best report (or aggregate) across slices
- Compute BLEU, CIDEr, ROUGE_L metrics

Data format:
- question_id: "dataset/patient_id/modality/images/slice_num"
- Each patient has multiple slices across modalities
"""

import json
import jsonlines
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import nltk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


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


def extract_patient_id(gt_entry: Dict) -> str:
    """Extract patient_id from gt_entry.

    Image format: "dataset/patient_id/modality/images/slice.png"
    """
    if 'image' in gt_entry:
        image_path = gt_entry['image']
        parts = image_path.split('/')
        if len(parts) >= 2:
            return parts[1]  # parts[0] is dataset, parts[1] is patient_id

    # Fallback: extract from question_id
    question_id = gt_entry['question_id']
    parts = question_id.split('/')
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def compute_metrics(gt_dict: Dict, pred_dict: Dict) -> Dict:
    """Compute BLEU, CIDEr, ROUGE_L, METEOR metrics"""
    if len(pred_dict) == 0:
        return {
            'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0,
            'CIDEr': 0.0, 'ROUGE_L': 0.0, 'METEOR': 0.0
        }

    evaluators = {
        'BLEU': Bleu(4),
        'CIDEr': Cider(),
        'ROUGE_L': Rouge()
    }

    results = {}

    for metric_name, evaluator in evaluators.items():
        try:
            score, _ = evaluator.compute_score(gt_dict, pred_dict)
            if metric_name == 'BLEU':
                results['BLEU-1'] = score[0]
                results['BLEU-2'] = score[1]
                results['BLEU-3'] = score[2]
                results['BLEU-4'] = score[3]
            else:
                results[metric_name] = score
        except Exception as e:
            print(f"Error computing {metric_name}: {e}")
            if metric_name == 'BLEU':
                results['BLEU-1'] = 0.0
                results['BLEU-2'] = 0.0
                results['BLEU-3'] = 0.0
                results['BLEU-4'] = 0.0
            else:
                results[metric_name] = 0.0

    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score as _nltk_meteor
        meteor_sum = 0.0
        for key in gt_dict:
            ref_tok = gt_dict[key][0].split()
            hyp_tok = pred_dict[key][0].split()
            meteor_sum += _nltk_meteor([ref_tok], hyp_tok)
        results['METEOR'] = meteor_sum / len(gt_dict)
    except Exception as e:
        print(f"Error computing METEOR: {e}")
        results['METEOR'] = 0.0

    return results


def evaluate_report_single(pred_file: str, gt_file: str):
    """Main evaluation function for single-modal report generation with patient-level aggregation"""

    print("=" * 70)
    print("Single-Image Report Generation Evaluation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")

    # Create mapping from question_id to gt
    gt_mapping = {gt['question_id']: gt for gt in ground_truth}

    # Organize data by patient
    data_by_patient = defaultdict(lambda: {'preds': [], 'gts': [], 'question_ids': []})

    for pred in predictions:
        question_id = pred['question_id']
        if question_id in gt_mapping:
            gt = gt_mapping[question_id]
            patient_id = extract_patient_id(gt)

            data_by_patient[patient_id]['preds'].append(pred['text'])
            data_by_patient[patient_id]['gts'].append(gt['label'])
            data_by_patient[patient_id]['question_ids'].append(question_id)

    # =========================================================================
    # Sample-Level Evaluation (All samples)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAMPLE-LEVEL PERFORMANCE")
    print("=" * 70)

    # Prepare data for sample-level evaluation
    sample_gt_dict = {}
    sample_pred_dict = {}

    for pred in predictions:
        question_id = pred['question_id']
        if question_id in gt_mapping:
            sample_gt_dict[question_id] = [gt_mapping[question_id]['label']]
            sample_pred_dict[question_id] = [pred['text']]

    sample_metrics = compute_metrics(sample_gt_dict, sample_pred_dict)

    print(f"\n{'Metric':<12} {'Score':<12}")
    print("-" * 30)
    for metric, score in sample_metrics.items():
        print(f"{metric:<12} {score:<12.4f}")
    print(f"\nTotal samples: {len(sample_pred_dict)}")

    # =========================================================================
    # Patient-Level Evaluation (Best report per patient)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PATIENT-LEVEL PERFORMANCE (Best Report Selection)")
    print("=" * 70)

    # For each patient, we use the first report as representative
    # (In practice, all slices of the same patient should have the same GT report)
    patient_gt_dict = {}
    patient_pred_dict = {}

    for patient_id, data in data_by_patient.items():
        # Use first sample as representative (GT should be same for all slices)
        patient_gt_dict[patient_id] = [data['gts'][0]]
        # Use first prediction (could also implement voting/selection strategy)
        patient_pred_dict[patient_id] = [data['preds'][0]]

    patient_metrics = compute_metrics(patient_gt_dict, patient_pred_dict)

    print(f"\n{'Metric':<12} {'Score':<12}")
    print("-" * 30)
    for metric, score in patient_metrics.items():
        print(f"{metric:<12} {score:<12.4f}")
    print(f"\nTotal patients: {len(patient_pred_dict)}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)

    return {
        'sample_level': sample_metrics,
        'patient_level': patient_metrics,
        'num_samples': len(sample_pred_dict),
        'num_patients': len(patient_pred_dict)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate single-modal report generation")
    parser.add_argument('--pred',
                        default="playground/coco/external1_single_report.json",
                        help='Path to predictions JSONL file')
    parser.add_argument('--gt',
                        default="/mnt/data/by/data/coco_new/labels_2025_11_29/external/test_report_generation_EN.json",
                        help='Path to ground truth JSON file')
    args = parser.parse_args()

    evaluate_report_single(args.pred, args.gt)
