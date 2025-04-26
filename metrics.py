#!/usr/bin/env python3
import numpy as np
import pandas as pd
import nibabel as nib
import json
import os
from monai.metrics import (
    DiceMetric, 
    HausdorffDistanceMetric,
    ConfusionMatrixMetric
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
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    cm_metric = ConfusionMatrixMetric(include_background=False, 
                                    metric_name=["sensitivity", "precision"],
                                    reduction="mean")
    
    results = []
    for case_id in tqdm(case_ids, desc="Calculating segmentation metrics"):
        try:
            # Load prediction and ground truth
            pred = nib.load(os.path.join(pred_dir, f"{case_id}.nii.gz")).get_fdata()
            gt = nib.load(os.path.join(gt_dir, f"{case_id}.nii.gz")).get_fdata()
            
            # Create binary masks for each structure
            pancreas_pred = (pred == 1).astype(np.float32)
            pancreas_gt = (gt == 1).astype(np.float32)
            
            lesion_pred = (pred == 2).astype(np.float32)
            lesion_gt = (gt == 2).astype(np.float32)
            
            # Combine for total foreground (pancreas + lesion)
            total_pred = ((pred == 1) | (pred == 2)).astype(np.float32)
            total_gt = ((gt == 1) | (gt == 2)).astype(np.float32)
            
            # Convert to torch tensors
            def to_tensor(arr):
                return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            
            # Calculate metrics for each structure
            metrics = {'case_id': case_id}
            
            for name, p, g in [('pancreas', pancreas_pred, pancreas_gt),
                              ('lesion', lesion_pred, lesion_gt),
                              ('total', total_pred, total_gt)]:
                
                p_t = to_tensor(p)
                g_t = to_tensor(g)
                
                # Calculate metrics
                dice_metric(p_t, g_t)
                hd_metric(p_t, g_t)
                cm_metric(p_t, g_t)
                cm_metric_list = cm_metric.aggregate()
                
                # Store results
                metrics[f'dice_{name}'] = dice_metric.aggregate().item()
                hd_value = hd_metric.aggregate().item()
                metrics[f'hd95_{name}'] = hd_value if not np.isinf(hd_value) else np.nan
                metrics[f'sensitivity_{name}'] = cm_metric_list[0].cpu().numpy()[0]
                metrics[f'precision_{name}'] = cm_metric_list[1].cpu().numpy()[0]
                
                # Reset metrics for next calculation
                dice_metric.reset()
                hd_metric.reset()
                cm_metric.reset()
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")
    
    return pd.DataFrame(results)

def calculate_classification_metrics(csv_path: str, logits_path: str) -> pd.DataFrame:
    """Calculate classification metrics from predicted logits"""
    # Load data
    df = pd.read_csv(csv_path)
    with open(logits_path) as f:
        logits_data = json.load(f)
    
    # Prepare case IDs and true labels
    case_ids = [os.path.basename(row['label']).replace('.nii.gz', '') 
               for _, row in df.iterrows()]
    true_labels = [row['Subtype'] for _, row in df.iterrows()]
    
    # Get predictions and probabilities
    pred_labels = []
    pred_probs = []
    for case_id in case_ids:
        probs = logits_data.get(case_id, [0.33, 0.33, 0.34])  # Uniform default if missing
        pred_probs.append(probs)
        pred_labels.append(np.argmax(probs))
    
    # Calculate metrics
    results = {
        'case_id': case_ids,
        'true_label': true_labels,
        'pred_label': pred_labels,
        'prob_class0': [p[0] for p in pred_probs],
        'prob_class1': [p[1] for p in pred_probs],
        'prob_class2': [p[2] for p in pred_probs],
        'correct': [t == p for t, p in zip(true_labels, pred_labels)]
    }
    
    return pd.DataFrame(results)

def plot_segmentation_metrics(seg_metrics: pd.DataFrame, output_dir: str):
    """Generate visualizations for segmentation metrics"""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    
    # Define consistent structure order and properties
    structures = ['pancreas', 'lesion', 'total']
    structure_labels = ['Pancreas', 'Lesion', 'Combined']
    palette = {'pancreas': '#1f77b4', 'lesion': '#ff7f0e', 'total': '#2ca02c'}
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten for easier iteration
    
    # Metrics to plot and their properties
    metrics = {
        'dice': {'title': 'Dice Score', 'ylim': (0, 1)},
        'hd95': {'title': 'HD95 Distance', 'ylim': None},
        'sensitivity': {'title': 'Sensitivity', 'ylim': (0, 1)},
        'precision': {'title': 'Precision', 'ylim': (0, 1)}
    }
    
    # Plot each metric
    for i, (metric, props) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Prepare data for this metric
        metric_cols = [f'{metric}_{s}' for s in structures]
        data = seg_metrics.melt(id_vars=['case_id'],
                              value_vars=metric_cols,
                              var_name='structure',
                              value_name='value')
        
        # Clean structure names
        data['structure'] = data['structure'].str.replace(f'{metric}_', '')
        
        # Ensure correct ordering
        data['structure'] = pd.Categorical(data['structure'], categories=structures, ordered=True)
        data = data.sort_values('structure')
        
        # Plot
        sns.boxplot(x='structure', y='value', data=data, ax=ax,
                   hue='structure', palette=palette, legend=False)
        sns.stripplot(x='structure', y='value', data=data, ax=ax,
                     color='black', alpha=0.5, order=structures)
        
        # Formatting
        ax.set_title(props['title'], fontsize=12)
        ax.set_ylabel(props['title'], fontsize=10)
        ax.set_xlabel('')
        ax.set_xticks(range(len(structures)))
        ax.set_xticklabels(structure_labels)
        
        if props['ylim']:
            ax.set_ylim(props['ylim'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segmentation_metrics.png'), dpi=300)
    plt.close()

def plot_classification_metrics(cls_metrics: pd.DataFrame, output_dir: str):
    """Generate visualizations for classification metrics without warnings"""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Ensure proper numeric types
    cls_metrics = cls_metrics.copy()
    cls_metrics['true_label'] = cls_metrics['true_label'].astype(int)
    cls_metrics['pred_label'] = cls_metrics['pred_label'].astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(cls_metrics['true_label'], cls_metrics['pred_label'])
    class_names = ['Class 0', 'Class 1', 'Class 2']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names,
                yticklabels=class_names)
    axes[0].set_title('Confusion Matrix', fontsize=12)
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('True', fontsize=10)
    
    # Probability distributions
    prob_df = cls_metrics.melt(
        id_vars=['case_id', 'true_label'], 
        value_vars=['prob_class0', 'prob_class1', 'prob_class2'],
        var_name='class', 
        value_name='probability'
    )
    prob_df['class'] = prob_df['class'].str.replace('prob_class', '').astype(int)
    prob_df['true_label'] = prob_df['true_label'].astype(int)
    
    # Plot with explicit numeric handling
    sns.boxplot(
        x='true_label', 
        y='probability', 
        hue='class', 
        data=prob_df, 
        palette='Set2', 
        ax=axes[1]
    )
    axes[1].set_title('Predicted Probabilities by True Class', fontsize=12)
    axes[1].set_ylabel('Probability', fontsize=10)
    axes[1].set_xlabel('True Class', fontsize=10)
    axes[1].legend(title='Predicted Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_metrics.png'), dpi=300)
    plt.close()


def save_reports(seg_metrics: pd.DataFrame, cls_metrics: pd.DataFrame, output_dir: str):
    """Save detailed metrics reports in CSV format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # SEGMENTATION REPORT
    structures = ['pancreas', 'lesion', 'total']
    metrics = ['dice', 'hd95', 'sensitivity', 'precision']
    
    seg_stats = {}
    for struct in structures:
        seg_stats[struct] = {}
        for metric in metrics:
            col = f'{metric}_{struct}'
            seg_stats[struct][f'{metric}_mean'] = seg_metrics[col].mean()
            seg_stats[struct][f'{metric}_std'] = seg_metrics[col].std()
    
    pd.DataFrame(seg_stats).T.to_csv(os.path.join(output_dir, 'segmentation_report.csv'))
    
    # CLASSIFICATION REPORT
    cls_metrics = cls_metrics.copy()
    cls_metrics['true_label'] = cls_metrics['true_label'].astype(int)
    cls_metrics['pred_label'] = cls_metrics['pred_label'].astype(int)
    
    cls_report = classification_report(
        cls_metrics['true_label'],
        cls_metrics['pred_label'],
        target_names=['Class 0', 'Class 1', 'Class 2'],
        output_dict=True,
        zero_division=0
    )
    
    # Convert report to DataFrame and save as CSV
    cls_report_df = pd.DataFrame(cls_report).transpose()
    
    # Add macro F1 and accuracy (which aren't included by default in classification_report)
    macro_f1 = f1_score(
        cls_metrics['true_label'],
        cls_metrics['pred_label'],
        average='macro',
        zero_division=0
    )
    accuracy = np.mean(cls_metrics['correct'])
    
    cls_report_df.loc['macro_avg', 'f1-score'] = macro_f1
    cls_report_df.loc['accuracy', 'f1-score'] = accuracy
    
    cls_report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Save raw data
    seg_metrics.to_csv(os.path.join(output_dir, 'segmentation_metrics.csv'), index=False)
    cls_metrics.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
    

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

    print("\nGenerating visualizations...")
    plot_segmentation_metrics(seg_metrics, args.output_dir)
    
    import pdb; pdb.set_trace()

    print("\nCalculating classification metrics...")
    cls_metrics = calculate_classification_metrics(args.csv_path, args.logits_path)
    print("\nGenerating visualizations...")
    plot_classification_metrics(cls_metrics, args.output_dir)
    
    print("\nSaving detailed reports...")
    save_reports(seg_metrics, cls_metrics, args.output_dir)
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()