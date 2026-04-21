import json
import jsonlines
from typing import List, Dict, Tuple
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


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


def calculate_similarity(pred_text: str, gt_text: str) -> Dict[str, float]:
    """Calculate multiple similarity metrics between prediction and ground truth"""

    # Tokenize
    pred_tokens = pred_text.lower().split()
    gt_tokens = gt_text.lower().split()

    # BLEU score (with smoothing for short sentences)
    smoothing = SmoothingFunction()
    bleu_score = sentence_bleu([gt_tokens], pred_tokens,
                               smoothing_function=smoothing.method1)

    # Sequence matcher similarity (character level)
    seq_similarity = SequenceMatcher(None, pred_text.lower(), gt_text.lower()).ratio()

    # Word overlap
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    if len(pred_set.union(gt_set)) > 0:
        word_overlap = len(pred_set.intersection(gt_set)) / len(pred_set.union(gt_set))
    else:
        word_overlap = 0.0

    # Average score
    avg_score = (bleu_score + seq_similarity + word_overlap) / 3

    return {
        'bleu': bleu_score,
        'seq_similarity': seq_similarity,
        'word_overlap': word_overlap,
        'avg_score': avg_score
    }


def select_best_examples(pred_file: str, gt_file: str, top_k: int = 10, min_length: int = 20):
    """Select best prediction examples based on similarity scores"""

    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    # Create mapping from question_id to ground truth
    gt_mapping = {gt['question_id']: gt for gt in ground_truth}

    # Calculate scores for all predictions
    print("Calculating similarity scores...")
    scored_examples = []

    for pred in predictions:
        question_id = pred['question_id']
        if question_id in gt_mapping:
            gt = gt_mapping[question_id]
            pred_text = pred['text']
            gt_text = gt['label']

            # Skip very short examples
            if len(pred_text.split()) < min_length or len(gt_text.split()) < min_length:
                continue

            scores = calculate_similarity(pred_text, gt_text)

            scored_examples.append({
                'question_id': question_id,
                'image': gt['image'],
                'prediction': pred_text,
                'ground_truth': gt_text,
                'scores': scores
            })

    # Sort by average score
    scored_examples.sort(key=lambda x: x['scores']['avg_score'], reverse=True)

    # Select top k
    top_examples = scored_examples[:top_k]

    print(f"\n{'='*80}")
    print(f"Top {top_k} Best Prediction Examples")
    print(f"{'='*80}\n")

    for i, example in enumerate(top_examples, 1):
        print(f"{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}")
        print(f"Question ID: {example['question_id']}")
        print(f"Image: {example['image']}")
        print(f"\nScores:")
        print(f"  - BLEU: {example['scores']['bleu']:.4f}")
        print(f"  - Sequence Similarity: {example['scores']['seq_similarity']:.4f}")
        print(f"  - Word Overlap: {example['scores']['word_overlap']:.4f}")
        print(f"  - Average Score: {example['scores']['avg_score']:.4f}")
        print(f"\nGround Truth:")
        print(f"  {example['ground_truth']}")
        print(f"\nPrediction:")
        print(f"  {example['prediction']}")
        print()

    return top_examples


def select_diverse_examples(pred_file: str, gt_file: str,
                           num_excellent: int = 3,
                           num_good: int = 3,
                           num_moderate: int = 2,
                           min_length: int = 20):
    """Select diverse examples across different quality levels"""

    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    # Create mapping from question_id to ground truth
    gt_mapping = {gt['question_id']: gt for gt in ground_truth}

    # Calculate scores for all predictions
    print("Calculating similarity scores...")
    scored_examples = []

    for pred in predictions:
        question_id = pred['question_id']
        if question_id in gt_mapping:
            gt = gt_mapping[question_id]
            pred_text = pred['text']
            gt_text = gt['label']

            # Skip very short examples
            if len(pred_text.split()) < min_length or len(gt_text.split()) < min_length:
                continue

            scores = calculate_similarity(pred_text, gt_text)

            scored_examples.append({
                'question_id': question_id,
                'image': gt['image'],
                'prediction': pred_text,
                'ground_truth': gt_text,
                'scores': scores
            })

    # Sort by average score
    scored_examples.sort(key=lambda x: x['scores']['avg_score'], reverse=True)

    # Select examples from different quality tiers
    total = len(scored_examples)

    # Excellent: top 10%
    excellent_examples = scored_examples[:max(1, total // 10)][:num_excellent]

    # Good: 10-30%
    good_start = max(1, total // 10)
    good_end = max(2, total * 3 // 10)
    good_examples = scored_examples[good_start:good_end][:num_good]

    # Moderate: 30-50%
    mod_start = max(2, total * 3 // 10)
    mod_end = max(3, total // 2)
    moderate_examples = scored_examples[mod_start:mod_end][:num_moderate]

    categories = [
        ("Excellent Predictions (Top 10%)", excellent_examples),
        ("Good Predictions (10-30%)", good_examples),
        ("Moderate Predictions (30-50%)", moderate_examples)
    ]

    print(f"\n{'='*80}")
    print(f"Diverse Example Selection")
    print(f"{'='*80}\n")

    for category_name, examples in categories:
        print(f"\n{'#'*80}")
        print(f"# {category_name}")
        print(f"{'#'*80}\n")

        for i, example in enumerate(examples, 1):
            print(f"{'-'*80}")
            print(f"Example {i}")
            print(f"{'-'*80}")
            print(f"Question ID: {example['question_id']}")
            print(f"Image: {example['image']}")
            print(f"\nScores:")
            print(f"  - BLEU: {example['scores']['bleu']:.4f}")
            print(f"  - Sequence Similarity: {example['scores']['seq_similarity']:.4f}")
            print(f"  - Word Overlap: {example['scores']['word_overlap']:.4f}")
            print(f"  - Average Score: {example['scores']['avg_score']:.4f}")
            print(f"\nGround Truth:")
            print(f"  {example['ground_truth']}")
            print(f"\nPrediction:")
            print(f"  {example['prediction']}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best prediction examples")
    parser.add_argument('--pred', default="playground/coco/totalv3_report.json",
                       help='Path to predictions JSONL file')
    parser.add_argument('--gt', default="/mnt/data/by/data/coco_new/labels/test_report_generation_EN.json",
                       help='Path to ground truth JSON file')
    parser.add_argument('--mode', choices=['best', 'diverse'], default='best',
                       help='Selection mode: best (top k) or diverse (across quality levels)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top examples to select (for best mode)')
    parser.add_argument('--min_length', type=int, default=20,
                       help='Minimum word length for examples')

    args = parser.parse_args()

    if args.mode == 'best':
        select_best_examples(args.pred, args.gt, top_k=args.top_k, min_length=args.min_length)
    else:
        select_diverse_examples(args.pred, args.gt, min_length=args.min_length)
