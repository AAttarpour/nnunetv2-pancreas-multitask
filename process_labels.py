"""
This code fixes the labels of the nnUNet pancreas dataset.
The labels are stored as float32, which is not correct.
The labels should be stored as uint8.
This code will convert the labels to uint8 and save them over the original files.
This code is not part of the nnUNet package.
It is a standalone script that can be used to fix the labels of the nnUNet pancreas dataset.
"""

import os
import nibabel as nib
import numpy as np

# Add all label folders you want to fix
label_dirs = [
    './nnUNet_raw/Dataset003_PancreasMultiTask/labelsTr',
    './nnUNet_raw/Dataset003_PancreasMultiTask/labelsVal',  
    './nnUNet_raw/Dataset003_PancreasMultiTask/labels', 
]

for labels_dir in label_dirs:
    if not os.path.isdir(labels_dir):
        print(f'Skipping {labels_dir} (not found)')
        continue

    print(f'\nProcessing {labels_dir}')
    for fname in os.listdir(labels_dir):
        if fname.endswith('.nii.gz'):
            path = os.path.join(labels_dir, fname)
            img = nib.load(path)
            data = img.get_fdata()

            # Round and convert to int
            corrected = np.rint(data).astype(np.uint8)

            # Save over original
            nib.save(nib.Nifti1Image(corrected, img.affine, img.header), path)
            print(f'Fixed: {fname}')
