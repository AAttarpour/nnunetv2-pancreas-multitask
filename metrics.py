#!/usr/bin/env python3
import numpy as np
import pandas as pd
import nibabel as nib
import json
import os
from monai.metrics import (
    DiceMetric, 
    HausdorffDistanceMetric
)
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch

def calculate_segmentation_metrics(pred_dir: str, gt_dir: str, csv_path: str) -> pd.DataFrame:
    """Calculate segmentation metrics for pancreas and lesion only (skip background)"""
    df = pd.read_csv(csv_path)
    case_ids = [os.path.basename(row['label']).replace('.nii.gz', '') 
               for _, row in df.iterrows()]
    
    # Initialize metrics
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    
    results = []
    for case_id in tqdm(case_ids, desc="Calculating segmentation metrics"):
        try:
            # Load prediction and ground truth
            pred = nib.load(os.path.join(pred_dir, f"{case_id}.nii.gz")).get_fdata()
            gt = nib.load(os.path.join(gt_dir, f"{case_id}.nii.gz")).get_fdata()
            
            # Create one-hot encoded tensors for pancreas (1) and lesion (2)
            pancreas_pred = (pred == 1).astype(np.float32)
            pancreas_gt = (gt == 1).astype(np.float32)
            
            lesion_pred = (pred == 2).astype(np.float32)
            lesion_gt = (gt == 2).astype(np.float32)
            
            # Stack as multi-channel tensors (pancreas=channel 0, lesion=channel 1)
            pred_t = torch.from_numpy(np.stack([pancreas_pred, lesion_pred])).unsqueeze(0).float()
            gt_t = torch.from_numpy(np.stack([pancreas_gt, lesion_gt])).unsqueeze(0).float()
            
            # Calculate metrics
            dice = dice_metric(pred_t, gt_t)
            hd = hd_metric(pred_t, gt_t)
            
            # Store results with anatomical names
            results.append({
                'case_id': case_id,
                'dice_pancreas': dice[0, 0].item(),
                'hd95_pancreas': hd[0, 0].item() if not torch.isinf(hd[0, 0]) else np.nan,
                'dice_lesion': dice[0, 1].item(),
                'hd95_lesion': hd[0, 1].item() if not torch.isinf(hd[0, 1]) else np.nan
            })
            
            # Reset metrics for next case
            dice_metric.reset()
            hd_metric.reset()
            
        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")
    
    return pd.DataFrame(results)

def plot_segmentation_metrics(seg_metrics: pd.DataFrame, output_dir: str):
    """Generate visualizations for segmentation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn')
    
    # Prepare data
    dice_df = seg_metrics.melt(id_vars=['case_id'], 
                             value_vars=['dice_pancreas', 'dice_lesion'],
                             var_name='structure', 
                             value_name='dice')
    
    hd_df = seg_metrics.melt(id_vars=['case_id'], 
                           value_vars=['hd95_pancreas', 'hd95_lesion'],
                           var_name='structure', 
                           value_name='hd95')
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Dice scores
    sns.boxplot(x='structure', y='dice', data=dice_df, ax=ax1, palette=['#1f77b4', '#ff7f0e'])
    sns.stripplot(x='structure', y='dice', data=dice_df, ax=ax1, color='black', alpha=0.5)
    ax1.set_title('Dice Scores by Anatomical Structure', fontsize=14)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticklabels(['Pancreas', 'Lesion'])
    ax1.set_ylim(0, 1)
    
    # HD95 distances
    sns.boxplot(x='structure', y='hd95', data=hd_df, ax=ax2, palette=['#1f77b4', '#ff7f0e'])
    sns.stripplot(x='structure', y='hd95', data=hd_df, ax=ax2, color='black', alpha=0.5)
    ax2.set_title('HD95 Distances by Anatomical Structure', fontsize=14)
    ax2.set_ylabel('HD95 (mm)', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_xticklabels(['Pancreas', 'Lesion'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anatomical_segmentation_metrics.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate segmentation and classification metrics')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predicted segmentations')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth segmentations')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to validation CSV file')
    parser.add_argument('--logits_path', type=str, required=True,
                       help='Path to classification logits JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("Calculating segmentation metrics...")
    seg_metrics = calculate_segmentation_metrics(args.pred_dir, args.gt_dir, args.csv_path)
    
    print("\nCalculating classification metrics...")
    cls_metrics = calculate_classification_metrics(args.csv_path, args.logits_path)
    
    print("\nGenerating visualizations...")
    plot_segmentation_metrics(seg_metrics, args.output_dir)
    plot_classification_metrics(cls_metrics, args.output_dir)
    
    print("\nSaving detailed reports...")
    save_reports(seg_metrics, cls_metrics, args.output_dir)
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()