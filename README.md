# nnunetv2-pancreas-multitask

# Requirements and installation:
conda env create -f conda_env.yml
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

# make the datasets structure ready for nnunetv2
python prepare_datasets_for_nnunetv2.py

# set the environment variables
export nnUNet_raw="./nnUNet_raw"
export nnUNet_preprocessed="./nnUNet_preprocessed"
export nnUNet_results="./nnUNet_results"

# add a classification head to the nnUNet architecture (nnUNetTrainerWithClassification.py)
cp nnUNetTrainerWithClassification.py ./nnUNet/nnunetv2/training/nnUNetTrainer/ 

# load the dataset for the pancreas multi-task segmentation task (seg and classification label)
cp pancreas_multitask_dataset.py ./nnUNet/nnunetv2/training/dataloading/custom_datasets/

# register dataset
add the following lines into the nnunet_dataset.py file
from nnunetv2.training.dataloading.custom_datasets.pancreas_multitask_dataset import PancreasMultiTaskDataset
file_ending_dataset_mapping['npz'] = PancreasMultiTaskDataset

# run the following script to map the labels from [0 1.0003 2] to [0 1 2]
python process_labels.py

# run the plan and preprocess
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity

# copy the .csv file containing the classification label into the nnUNet_proprocessed (created with the above command)
cp ./nnUNet_raw/Dataset003_PancreasMultiTask/subtype_results_train.csv ./nnUNet_preprocessed/Dataset003_PancreasMultiTask


