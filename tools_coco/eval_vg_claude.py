import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from typing import List, Tuple, Dict, Any, Optional

def parse_bounding_box(text: str) -> Optional[List[int]]:
    """
    Extract bounding box coordinates from text containing <box>[[x1, y1, x2, y2]]</box> format.
    
    Args:
        text: Text containing bounding box annotation
        
    Returns:
        List of [x1, y1, x2, y2] coordinates, or None if not found
    """
    pattern = r'<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>'
    match = re.search(pattern, text)
    
    if match:
        return [int(match.group(1)), int(match.group(2)), 
                int(match.group(3)), int(match.group(4))]
    return None

def denormalize_bbox(bbox: List[int], img_width: int, img_height: int) -> List[int]:
    """
    Convert bbox from 0-1000 scale to actual image coordinates.
    
    Args:
        bbox: [x1, y1, x2, y2] in 0-1000 scale
        img_width: Actual image width
        img_height: Actual image height
        
    Returns:
        [x1, y1, x2, y2] in actual image coordinates
    """
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    # Convert from 0-1000 scale to actual image coordinates
    actual_x1 = int(x1 * img_width / 1000)
    actual_y1 = int(y1 * img_height / 1000)
    actual_x2 = int(x2 * img_width / 1000)
    actual_y2 = int(y2 * img_height / 1000)
    
    return [actual_x1, actual_y1, actual_x2, actual_y2]

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU score between 0 and 1
    """
    if box1 is None or box2 is None:
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
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union

def load_predictions(jsonl_path: str) -> Dict[str, str]:
    """
    Load predictions from JSONL file.
    
    Args:
        jsonl_path: Path to predictions JSONL file
        
    Returns:
        Dictionary mapping question_id to prediction text
    """
    predictions = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predictions[data['question_id']] = data['text']
    return predictions

def load_ground_truth(json_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load ground truth from JSON file.
    
    Args:
        json_path: Path to ground truth JSON file
        
    Returns:
        Tuple of (ground_truth_labels, modalities) dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    ground_truth = {}
    modalities = {}
    for item in gt_data:
        ground_truth[item['question_id']] = item['label']
        modalities[item['question_id']] = item['modality']
    
    return ground_truth, modalities

def evaluate_visual_grounding(pred_jsonl_path: str, gt_json_path: str) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluate visual grounding performance with bounding box predictions.
    
    Args:
        pred_jsonl_path: Path to predictions JSONL file
        gt_json_path: Path to ground truth JSON file
        
    Returns:
        Tuple of (overall_metrics, modality_metrics)
    """
    # Load data
    predictions = load_predictions(pred_jsonl_path)
    ground_truth, modalities = load_ground_truth(gt_json_path)
    
    # Find common question IDs
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    
    if not common_ids:
        raise ValueError("No matching question IDs found between predictions and ground truth")
    
    print(f"Evaluating {len(common_ids)} samples...")
    
    # Initialize metrics tracking
    all_ious = []
    iou_thresholds = [0.1, 0.3, 0.5]
    threshold_counts = {thresh: 0 for thresh in iou_thresholds}
    valid_count = 0
    
    # Modality-specific tracking
    modality_stats = {}
    
    for question_id in common_ids:
        pred_text = predictions[question_id]
        gt_text = ground_truth[question_id]
        modality = modalities[question_id]
        
        # Initialize modality tracking
        if modality not in modality_stats:
            modality_stats[modality] = {
                'ious': [],
                'threshold_counts': {thresh: 0 for thresh in iou_thresholds},
                'valid_count': 0,
                'total_count': 0
            }
        
        modality_stats[modality]['total_count'] += 1
        
        # Parse bounding boxes
        pred_box = parse_bounding_box(pred_text)
        gt_box = parse_bounding_box(gt_text)
        
        if pred_box is None:
            print(f"Warning: Failed to parse prediction box for {question_id}")
            continue
            
        if gt_box is None:
            print(f"Warning: Failed to parse ground truth box for {question_id}")
            continue
        
        # Calculate IoU
        iou_score = calculate_iou(pred_box, gt_box)
        
        # Update overall metrics
        all_ious.append(iou_score)
        valid_count += 1
        
        for thresh in iou_thresholds:
            if iou_score >= thresh:
                threshold_counts[thresh] += 1
        
        # Update modality-specific metrics
        modality_stats[modality]['ious'].append(iou_score)
        modality_stats[modality]['valid_count'] += 1
        
        for thresh in iou_thresholds:
            if iou_score >= thresh:
                modality_stats[modality]['threshold_counts'][thresh] += 1
    
    if valid_count == 0:
        raise ValueError("No valid predictions found")
    
    # Calculate overall results
    overall_results = {
        'mIoU': np.mean(all_ious),
        'total_samples': len(common_ids),
        'valid_predictions': valid_count,
        'invalid_predictions': len(common_ids) - valid_count
    }
    
    for thresh in iou_thresholds:
        overall_results[f'IoU@{thresh}'] = threshold_counts[thresh] / valid_count
    
    # Calculate modality-specific results
    modality_results = {}
    for modality, stats in modality_stats.items():
        if stats['valid_count'] > 0:
            modality_results[modality] = {
                'mIoU': np.mean(stats['ious']),
                'total_samples': stats['total_count'],
                'valid_predictions': stats['valid_count'],
                'invalid_predictions': stats['total_count'] - stats['valid_count']
            }
            
            for thresh in iou_thresholds:
                modality_results[modality][f'IoU@{thresh}'] = stats['threshold_counts'][thresh] / stats['valid_count']
        else:
            modality_results[modality] = {
                'mIoU': 0.0,
                'total_samples': stats['total_count'],
                'valid_predictions': 0,
                'invalid_predictions': stats['total_count']
            }
            
            for thresh in iou_thresholds:
                modality_results[modality][f'IoU@{thresh}'] = 0.0
    
    return overall_results, modality_results

def display_results(overall_results: Dict[str, float], modality_results: Dict[str, Dict[str, float]]) -> None:
    """Display evaluation results in a formatted table."""
    print("\n" + "="*80)
    print("VISUAL GROUNDING EVALUATION RESULTS")
    print("="*80)
    
    # Overall performance
    print("OVERALL PERFORMANCE:")
    print("-"*80)
    print(f"Total samples: {overall_results['total_samples']}")
    print(f"Valid predictions: {overall_results['valid_predictions']}")
    print(f"Invalid predictions: {overall_results['invalid_predictions']}")
    print(f"mIoU (mean IoU): {overall_results['mIoU']:.4f}")
    print(f"IoU@0.1: {overall_results['IoU@0.1']:.4f} ({overall_results['IoU@0.1']*100:.1f}%)")
    print(f"IoU@0.3: {overall_results['IoU@0.3']:.4f} ({overall_results['IoU@0.3']*100:.1f}%)")
    print(f"IoU@0.5: {overall_results['IoU@0.5']:.4f} ({overall_results['IoU@0.5']*100:.1f}%)")
    
    # Modality breakdown
    print("\nPERFORMANCE BY MODALITY:")
    print("-"*80)
    
    for modality in sorted(modality_results.keys()):
        results = modality_results[modality]
        print(f"\nModality: {modality}")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Valid predictions: {results['valid_predictions']}")
        print(f"  Invalid predictions: {results['invalid_predictions']}")
        print(f"  mIoU: {results['mIoU']:.4f}")
        print(f"  IoU@0.1: {results['IoU@0.1']:.4f} ({results['IoU@0.1']*100:.1f}%)")
        print(f"  IoU@0.3: {results['IoU@0.3']:.4f} ({results['IoU@0.3']*100:.1f}%)")
        print(f"  IoU@0.5: {results['IoU@0.5']:.4f} ({results['IoU@0.5']*100:.1f}%)")
    
    # Summary table
    print("\nSUMMARY TABLE:")
    print("-"*80)
    header = f"{'Modality':<15} {'Samples':<8} {'Valid':<6} {'mIoU':<8} {'IoU@0.1':<8} {'IoU@0.3':<8} {'IoU@0.5':<8}"
    print(header)
    print("-"*80)
    
    # Overall row
    print(f"{'Overall':<15} {overall_results['total_samples']:<8} {overall_results['valid_predictions']:<6} "
          f"{overall_results['mIoU']:<8.4f} {overall_results['IoU@0.1']:<8.4f} "
          f"{overall_results['IoU@0.3']:<8.4f} {overall_results['IoU@0.5']:<8.4f}")
    
    # Modality rows
    for modality in sorted(modality_results.keys()):
        results = modality_results[modality]
        print(f"{modality:<15} {results['total_samples']:<8} {results['valid_predictions']:<6} "
              f"{results['mIoU']:<8.4f} {results['IoU@0.1']:<8.4f} "
              f"{results['IoU@0.3']:<8.4f} {results['IoU@0.5']:<8.4f}")
    
    print("="*80)

def visualize_predictions(pred_jsonl_path: str, gt_json_path: str, data_root: str, output_dir: str = "visualization_results"):
    """
    Visualize best and worst predictions for each modality.
    
    Args:
        pred_jsonl_path: Path to predictions JSONL file
        gt_json_path: Path to ground truth JSON file
        data_root: Root directory containing images
        output_dir: Output directory for visualization results
    """
    # Load data and calculate IoUs
    predictions = load_predictions(pred_jsonl_path)
    ground_truth, modalities = load_ground_truth(gt_json_path)
    
    # Load images info from ground truth
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    image_paths = {item['question_id']: item['image'] for item in gt_data}
    
    common_ids = set(predictions.keys()) & set(ground_truth.keys())
    
    # Calculate IoUs for all samples
    sample_ious = {}
    modality_samples = {}
    
    for question_id in common_ids:
        pred_text = predictions[question_id]
        gt_text = ground_truth[question_id]
        modality = modalities[question_id]
        
        # Initialize modality tracking
        if modality not in modality_samples:
            modality_samples[modality] = []
        
        # Parse bounding boxes
        pred_box = parse_bounding_box(pred_text)
        gt_box = parse_bounding_box(gt_text)
        
        if pred_box is None or gt_box is None:
            continue
        
        # Calculate IoU
        iou_score = calculate_iou(pred_box, gt_box)
        
        sample_ious[question_id] = {
            'iou': iou_score,
            'pred_box': pred_box,
            'gt_box': gt_box,
            'modality': modality,
            'image_path': image_paths[question_id],
            'pred_text': pred_text,
            'gt_text': gt_text
        }
        
        modality_samples[modality].append(question_id)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # For each modality, select best 10 and worst 10
    for modality in modality_samples.keys():
        modality_ids = modality_samples[modality]
        modality_data = [(qid, sample_ious[qid]['iou']) for qid in modality_ids]
        
        # Sort by IoU
        modality_data.sort(key=lambda x: x[1])
        
        # Select worst 10 and best 10
        worst_10 = modality_data[:10]
        best_10 = modality_data[-10:]
        
        print(f"\nProcessing {modality} modality:")
        print(f"  Total samples: {len(modality_data)}")
        print(f"  Best IoU: {best_10[-1][1]:.4f}")
        print(f"  Worst IoU: {worst_10[0][1]:.4f}")
        
        # Visualize worst 10
        visualize_sample_set(worst_10, sample_ious, data_root, 
                           output_dir, f"{modality}_worst", "Worst Predictions")
        
        # Visualize best 10
        visualize_sample_set(best_10, sample_ious, data_root, 
                           output_dir, f"{modality}_best", "Best Predictions")

def visualize_sample_set(sample_list: List[Tuple[str, float]], sample_ious: Dict, 
                        data_root: str, output_dir: str, prefix: str, title_prefix: str):
    """
    Visualize a set of samples with their bounding boxes.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"{title_prefix} - {prefix.replace('_', ' ').title()}", fontsize=16)
    
    for idx, (question_id, iou_score) in enumerate(sample_list):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        sample_data = sample_ious[question_id]
        image_path = os.path.join(data_root, sample_data['image_path'])
        
        try:
            # Load image
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # Convert bboxes to actual coordinates
            pred_box_actual = denormalize_bbox(sample_data['pred_box'], img_width, img_height)
            gt_box_actual = denormalize_bbox(sample_data['gt_box'], img_width, img_height)
            
            # Display image
            ax.imshow(img, cmap='gray' if len(np.array(img).shape) == 2 else None)
            
            # Draw ground truth box (green)
            if gt_box_actual:
                x1, y1, x2, y2 = gt_box_actual
                rect_gt = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          linewidth=2, edgecolor='green', 
                                          facecolor='none', label='Ground Truth')
                ax.add_patch(rect_gt)
            
            # Draw prediction box (red)
            if pred_box_actual:
                x1, y1, x2, y2 = pred_box_actual
                rect_pred = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            linewidth=2, edgecolor='red', 
                                            facecolor='none', label='Prediction')
                ax.add_patch(rect_pred)
            
            # Set title with IoU score
            ax.set_title(f'IoU: {iou_score:.3f}\n{os.path.basename(image_path)}', fontsize=10)
            ax.axis('off')
            
            # Add legend only for first subplot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{os.path.basename(image_path)}\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'IoU: {iou_score:.3f}', fontsize=10)
            ax.axis('off')
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{prefix}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {output_path}")

def main():
    """Main evaluation function with example usage."""
    # Configuration
    # pred_jsonl_path = "playground/coco/total_full_vg.json"
    # gt_json_path = "/mnt/data/by/data/coco_new/labels/ct_test_bbox.json"
    # data_root = "/mnt/data/by/data/coco_new/data"

    # external
    # pred_jsonl_path = "playground/coco/external_lora_vg.json"
    # gt_json_path = "/mnt/data/by/data/coco_new/label_external_final_v2/ct_external_non_filter_bbox.json"


    # internal    
    pred_jsonl_path = "playground/coco/total_full_vg.json"
    gt_json_path = "/mnt/data/by/data/coco_new/labels/ct_test_bbox_v2.json.bak"
    data_root = "/mnt/data/by/data/coco_new"
    
    try:
        # Run evaluation
        overall_results, modality_results = evaluate_visual_grounding(pred_jsonl_path, gt_json_path)
        
        # Display results
        display_results(overall_results, modality_results)
        
        # Quick summary
        print(f"\nQUICK SUMMARY:")
        print(f"Overall mIoU: {overall_results['mIoU']:.4f}")
        for modality in sorted(modality_results.keys()):
            print(f"{modality} mIoU: {modality_results[modality]['mIoU']:.4f}")
        
        # Generate visualizations
        print(f"\nGenerating visualizations...")
        visualize_predictions(pred_jsonl_path, gt_json_path, data_root)
        print(f"Visualization complete! Check 'visualization_results' directory.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please verify the file paths are correct.")
    except Exception as e:
        print(f"Evaluation error: {e}")

if __name__ == "__main__":
    main()