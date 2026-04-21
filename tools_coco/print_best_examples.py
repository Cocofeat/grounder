import json
import jsonlines
from typing import List, Dict
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


def print_copyable_examples(pred_file: str, gt_file: str, num_examples: int = 5, min_length: int = 20):
    """Print examples in a clean, copyable format"""

    print("Loading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    # Create mapping from question_id to ground truth
    gt_mapping = {gt['question_id']: gt for gt in ground_truth}

    # Calculate scores for all predictions
    print("Calculating similarity scores...\n")
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

    # Print in copyable format
    print("="*100)
    print(f"优秀预测示例 (Top {num_examples})")
    print("="*100)
    print()

    for i in range(min(num_examples, len(scored_examples))):
        example = scored_examples[i]

        print(f"【示例 {i+1}】")
        print(f"Question ID: {example['question_id']}")
        print(f"Image: {example['image']}")
        print(f"相似度得分: BLEU={example['scores']['bleu']:.4f}, Seq={example['scores']['seq_similarity']:.4f}, WordOverlap={example['scores']['word_overlap']:.4f}, Avg={example['scores']['avg_score']:.4f}")
        print()
        print("Ground Truth:")
        print(example['ground_truth'])
        print()
        print("Prediction:")
        print(example['prediction'])
        print()
        print("-"*100)
        print()

    # Also print some good but not perfect examples
    print("\n")
    print("="*100)
    print("良好预测示例 (中等质量，更有代表性)")
    print("="*100)
    print()

    # Get examples from 10-30% range
    start_idx = max(1, len(scored_examples) // 10)
    end_idx = max(2, len(scored_examples) * 3 // 10)
    good_examples = scored_examples[start_idx:end_idx][:3]

    for i, example in enumerate(good_examples, 1):
        print(f"【示例 {i}】")
        print(f"Question ID: {example['question_id']}")
        print(f"Image: {example['image']}")
        print(f"相似度得分: BLEU={example['scores']['bleu']:.4f}, Seq={example['scores']['seq_similarity']:.4f}, WordOverlap={example['scores']['word_overlap']:.4f}, Avg={example['scores']['avg_score']:.4f}")
        print()
        print("Ground Truth:")
        print(example['ground_truth'])
        print()
        print("Prediction:")
        print(example['prediction'])
        print()
        print("-"*100)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print best prediction examples in copyable format")
    parser.add_argument('--pred', default="playground/coco/totalv3_report.json",
                       help='Path to predictions JSONL file')
    parser.add_argument('--gt', default="/mnt/data/by/data/coco_new/labels/test_report_generation_EN.json",
                       help='Path to ground truth JSON file')
    parser.add_argument('--num', type=int, default=3,
                       help='Number of top examples to print')
    parser.add_argument('--min_length', type=int, default=20,
                       help='Minimum word length for examples')

    args = parser.parse_args()

    print_copyable_examples(args.pred, args.gt, num_examples=args.num, min_length=args.min_length)
