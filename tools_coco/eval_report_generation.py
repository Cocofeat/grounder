import json
import jsonlines
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import nltk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.meteor.meteor import Meteor  # Removed: requires Java and can hang

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


def parse_image_path(image_path: str) -> Tuple[str, str, str]:
    """Parse image path to extract patient_id, modality, and instance"""
    # Supports both legacy paths like "PATIENT/MOD/images/idx.png"
    # and dataset-prefixed paths like "dataset/PATIENT/MOD/images/idx.png"
    parts = image_path.split('/')
    if 'images' in parts:
        idx = parts.index('images')
        patient_idx = max(idx - 2, 0)
        patient_id = parts[patient_idx]
        modality_idx = max(idx - 1, 0)
        modality = parts[modality_idx]
        instance = parts[idx + 1] if idx + 1 < len(parts) else parts[-1]
    else:
        patient_id = parts[0] if parts else ''
        modality = parts[1] if len(parts) > 1 else ''
        instance = parts[-1] if parts else ''
    return patient_id, modality, instance


def prepare_data_for_evaluation(predictions: List[Dict], ground_truth: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Prepare data in the format required by pycocoevalcap
    Returns: (ground_truth_dict, predictions_dict)
    Each dict maps question_id to list of texts
    """
    gt_dict = {}
    pred_dict = {}
    
    # Create mapping from question_id to ground truth
    gt_mapping = {gt['question_id']: gt for gt in ground_truth}
    
    for pred in predictions:
        question_id = pred['question_id']
        if question_id in gt_mapping:
            gt_dict[question_id] = [gt_mapping[question_id]['label']]
            pred_dict[question_id] = [pred['text']]
        else:
            print(f"Warning: Question ID {question_id} not found in ground truth")
    
    return gt_dict, pred_dict


def evaluate_report_generation(pred_file: str, gt_file: str, output_file: str = None):
    """Main evaluation function for report generation with BLEU and CIDEr metrics"""
    
    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)
    
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")
    
    # Prepare data for evaluation
    gt_dict, pred_dict = prepare_data_for_evaluation(predictions, ground_truth)
    
    print(f"Matched {len(pred_dict)} predictions with ground truth")
    
    if len(pred_dict) == 0:
        print("No matching predictions found!")
        return
    
    # Initialize evaluators
    print("\nComputing metrics...")
    evaluators = {
        'BLEU': Bleu(4),  # BLEU-1, BLEU-2, BLEU-3, BLEU-4
        'CIDEr': Cider(),
        'ROUGE_L': Rouge()
    }
    # Note: METEOR removed as it requires Java and can cause hanging issues
    
    results = {}
    
    # Compute each metric
    for metric_name, evaluator in evaluators.items():
        print(f"Computing {metric_name}...")
        try:
            score, scores = evaluator.compute_score(gt_dict, pred_dict)
            results[metric_name] = {
                'score': score,
                'scores': scores
            }
            
            if metric_name == 'BLEU':
                print(f"BLEU-1: {score[0]:.4f}")
                print(f"BLEU-2: {score[1]:.4f}")
                print(f"BLEU-3: {score[2]:.4f}")
                print(f"BLEU-4: {score[3]:.4f}")
            else:
                print(f"{metric_name}: {score:.4f}")
        except Exception as e:
            print(f"Error computing {metric_name}: {e}")
            results[metric_name] = None
    
    # Organize data by different dimensions for detailed analysis
    data_by_modality = defaultdict(lambda: {'pred': [], 'gt': [], 'question_ids': []})
    data_by_patient = defaultdict(lambda: {'pred': [], 'gt': [], 'question_ids': []})
    
    # Create mapping for detailed analysis
    gt_mapping = {gt['question_id']: gt for gt in ground_truth}
    
    for pred in predictions:
        question_id = pred['question_id']
        if question_id in gt_mapping:
            gt = gt_mapping[question_id]
            
            # Parse patient info
            patient_id, modality, instance = parse_image_path(gt['image'])
            
            # Store by modality
            data_by_modality[modality]['pred'].append(pred['text'])
            data_by_modality[modality]['gt'].append(gt['label'])
            data_by_modality[modality]['question_ids'].append(question_id)
            
            # Store by patient
            data_by_patient[patient_id]['pred'].append(pred['text'])
            data_by_patient[patient_id]['gt'].append(gt['label'])
            data_by_patient[patient_id]['question_ids'].append(question_id)
    
    # Evaluate by modality
    print("\n=== Performance by Modality ===")
    modality_results = {}
    for modality, data in data_by_modality.items():
        if len(data['pred']) > 0:
            # Prepare data for this modality
            mod_gt_dict = {}
            mod_pred_dict = {}
            for i, qid in enumerate(data['question_ids']):
                mod_gt_dict[qid] = [data['gt'][i]]
                mod_pred_dict[qid] = [data['pred'][i]]
            
            mod_results = {}
            for metric_name, evaluator in evaluators.items():
                try:
                    score, _ = evaluator.compute_score(mod_gt_dict, mod_pred_dict)
                    if metric_name == 'BLEU':
                        mod_results[metric_name] = score  # All BLEU scores
                        print(f"{modality} - BLEU-4: {score[3]:.4f} ({len(data['pred'])} samples)")
                    else:
                        mod_results[metric_name] = score
                        print(f"{modality} - {metric_name}: {score:.4f} ({len(data['pred'])} samples)")
                except Exception as e:
                    print(f"Error computing {metric_name} for {modality}: {e}")
                    mod_results[metric_name] = None
            
            modality_results[modality] = mod_results
    
    # Create summary results
    summary = {
        'overall_metrics': results,
        'modality_metrics': modality_results,
        'dataset_stats': {
            'total_predictions': len(predictions),
            'total_ground_truth': len(ground_truth),
            'matched_samples': len(pred_dict),
            'modalities': list(data_by_modality.keys()),
            'unique_patients': len(data_by_patient)
        }
    }
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total samples evaluated: {len(pred_dict)}")
    print(f"Number of modalities: {len(data_by_modality)}")
    print(f"Number of unique patients: {len(data_by_patient)}")
    
    # Save results if output file is specified
    if output_file:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj
        
        # Deep convert all numpy arrays
        import numpy as np
        def deep_convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return obj
        
        summary_serializable = deep_convert(summary)
        
        with open(output_file, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate report generation with BLEU and CIDEr metrics")
    parser.add_argument('--pred', default="playground/coco/total_full_report.json", 
                       help='Path to predictions JSONL file')
    parser.add_argument('--gt', default="/mnt/data/by/data/coco_new/labels/test_report_generation_EN.json", 
                       help='Path to ground truth JSON file')
    parser.add_argument('--output', default="tools_coco/report_generation_evaluation_results.json",
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_report_generation(args.pred, args.gt, args.output)
