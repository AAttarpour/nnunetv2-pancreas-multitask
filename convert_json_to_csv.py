import json
import pandas as pd
import os

def convert_logits_to_subtype_csv(json_path: str, output_csv_path: str):
    # Load the classification logits
    with open(json_path, 'r') as f:
        logits_data = json.load(f)
    
    # Prepare rows
    rows = []
    for case_id, probs in logits_data.items():
        pred_label = int(pd.Series(probs).idxmax())  # Find the index of max prob
        filename = f"{case_id}.nii.gz"  # Add .nii.gz as requested
        rows.append({'Names': filename, 'Subtype': pred_label})
    
    # Create a DataFrame
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Saved subtype predictions to {output_csv_path}")

if __name__ == "__main__":
    input_json = "nnUNet_raw/Dataset003_PancreasMultiTask/predictionsTs/classification_logits.json"  # <-- your logits json file
    output_csv = "nnUNet_raw/Dataset003_PancreasMultiTask/predictionsTs/subtype_results.csv"
    convert_logits_to_subtype_csv(input_json, output_csv)
