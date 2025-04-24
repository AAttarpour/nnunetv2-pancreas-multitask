"""
This script is used to prepare the datasets for nnunetv2; according to their webpage:
Example tree structure:
nnUNet_raw/Dataset001_NAME1
├── dataset.json
├── imagesTr
│   ├── ...
├── imagesTs
│   ├── ...
└── labelsTr
    ├── ...

inputs:
- original_dataset_path: Path to the original dataset.
- nnunet_raw_path: Path to the nnUNet raw dataset folder.
- dataset_id: Dataset ID for nnUNet.
- dataset_name: Name of the dataset.

outputs:
- dataset.json: JSON file containing dataset metadata.
- imagesTr: Directory containing training images.
- labelsTr: Directory containing training labels.
- imagesTs: Directory containing test images.
- imagesVal: Directory containing validation images.
- labelsVal: Directory containing validation labels.
- subtype_results_train.csv: CSV file containing classification labels for training set: Label, Subtype.
- validation_set.csv: CSV file containing validation tracking information: Image, Label, Subtype.
"""

import os
import shutil
import json
import pandas as pd

# === CONFIG ===
original_dataset_path = "./datasets"
nnunet_raw_path = "./nnUNet_raw"
dataset_id = 3
dataset_name = "PancreasMultiTask"
# ====================

dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
output_path = os.path.join(nnunet_raw_path, dataset_folder)
imagesTr_path = os.path.join(output_path, "imagesTr")
labelsTr_path = os.path.join(output_path, "labelsTr")
imagesVal_path = os.path.join(output_path, "imagesVal")
labelsVal_path = os.path.join(output_path, "labelsVal")
imagesTs_path = os.path.join(output_path, "imagesTs")

# Create output directories if they don't exist
os.makedirs(imagesTr_path, exist_ok=True)
os.makedirs(labelsTr_path, exist_ok=True)
os.makedirs(imagesVal_path, exist_ok=True)
os.makedirs(labelsVal_path, exist_ok=True)
os.makedirs(imagesTs_path, exist_ok=True)

training_entries = []
test_entries = []
classification_labels = []
validation_entries = []

counter_train = 0
counter_val = 0

# ---------- Process training data only
for subtype in ["subtype0", "subtype1", "subtype2"]:
    subtype_dir = os.path.join(original_dataset_path, "train", subtype)
    subtype_id = int(subtype[-1])
    for file in os.listdir(subtype_dir):
        if file.endswith("_0000.nii.gz"):
            base = file.replace("_0000.nii.gz", "")
            new_base = f"quiz_{counter_train:03d}"
            # Copy image
            shutil.copy(os.path.join(subtype_dir, file), os.path.join(imagesTr_path, f"{new_base}_0000.nii.gz"))
            # Copy label
            shutil.copy(os.path.join(subtype_dir, f"{base}.nii.gz"), os.path.join(labelsTr_path, f"{new_base}.nii.gz"))
            # Log entry
            training_entries.append({
                "image": f"./imagesTr/{new_base}_0000.nii.gz",
                "label": f"./labelsTr/{new_base}.nii.gz"
            })
            classification_labels.append({
                "Name": f"{new_base}.nii.gz",
                "Subtype": subtype_id
            })
            counter_train += 1

# Process validation data separately
for subtype in ["subtype0", "subtype1", "subtype2"]:
    subtype_dir = os.path.join(original_dataset_path, "validation", subtype)
    subtype_id = int(subtype[-1])
    for file in os.listdir(subtype_dir):
        if file.endswith("_0000.nii.gz"):
            base = file.replace("_0000.nii.gz", "")
            new_base = f"val_{counter_val:03d}"
            # Copy image and label
            shutil.copy(os.path.join(subtype_dir, file), os.path.join(imagesVal_path, f"{new_base}_0000.nii.gz"))
            shutil.copy(os.path.join(subtype_dir, f"{base}.nii.gz"), os.path.join(labelsVal_path, f"{new_base}.nii.gz"))
            validation_entries.append({
                "image": f"./imagesVal/{new_base}_0000.nii.gz",
                "label": f"./labelsVal/{new_base}.nii.gz",
                "Subtype": subtype_id
            })
            counter_val += 1

# Copy test set
test_dir = os.path.join(original_dataset_path, "test")
for file in sorted(os.listdir(test_dir)):
    if file.endswith("_0000.nii.gz"):
        shutil.copy(os.path.join(test_dir, file), os.path.join(imagesTs_path, file))
        test_entries.append(f"./imagesTs/{file}")

# Save dataset.json (train + test only)
# dataset_json = {
#     "name": dataset_name,
#     "description": "Pancreas segmentation and subtype classification",
#     "tensorImageSize": "3D",
#     "modality": {"0": "CT"},
#     "labels": {"0": "background", "1": "pancreas", "2": "lesion"},
#     "numTraining": len(training_entries),
#     "file_ending": ".nii.gz",
#     "training": training_entries,
#     "test": test_entries
# }
dataset_json = {
    "channel_names": {
        "0": "CT"
    },
    "labels": {  # THIS IS DIFFERENT NOW!
        "background": 0,
        "pancreas": 1,
        "lesion": 2
    }, 
    "file_ending": ".nii.gz",
    "numTraining": len(training_entries),
}

with open(os.path.join(output_path, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

# Save classification labels for training set
pd.DataFrame(classification_labels).to_csv(os.path.join(output_path, "subtype_results_train.csv"), index=False)

# Save validation tracking CSV
pd.DataFrame(validation_entries).to_csv(os.path.join(output_path, "validation_set.csv"), index=False)