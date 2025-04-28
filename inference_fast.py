"""
This script is a modified version of the nnUNetPredictor class from nnunetv2.
It's the same is the inference.py script, but with the addition of some techniques such as automatic mixed precision
(FP16) inference and a patch to the nnUNetTrainerWithClassification class to build the network architecture with a classification head.

"""

import torch
import numpy as np
import os
import json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.utilities.helpers import empty_cache
import json
import multiprocessing
import os
from time import sleep, time
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerWithClassification import nnUNetTrainerWithClassification, ClassificationHead
from queue import Queue
from threading import Thread
from tqdm import tqdm
from typing import Tuple, Union, List, Optional

class PredictorWithClassification(nnUNetPredictor):
     


    # def initialize_from_trained_model_folder(self, model_training_output_dir: str,
    #                                    use_folds: Union[Tuple[Union[int, str]], None],
    #                                    checkpoint_name: str = 'checkpoint_final.pth'):
    #     """
    #     Modified initialization to enable FP16 inference while keeping certain layers in FP32
    #     for numerical stability.
    #     """
    #     # First let the parent class do its normal initialization
    #     super().initialize_from_trained_model_folder(
    #         model_training_output_dir=model_training_output_dir,
    #         use_folds=use_folds,
    #         checkpoint_name=checkpoint_name
    #     )
        
    #     # Convert model to FP16
    #     print("Converting model to mixed precision (FP16) for faster inference")
    #     self.network = self.network.half()
        
    #     # Ensure certain layers stay in FP32 for numerical stability
    #     # def _convert_bn_and_head_to_float(module):
    #     #     if isinstance(module, (torch.nn.BatchNorm1d, 
    #     #                          torch.nn.BatchNorm3d,
    #     #                          ClassificationHead)):
    #     #         return module.float()
    #     #     return module
            
    #     # self.network = self.network.apply(_convert_bn_and_head_to_float)
    #     self.network = self.network.to(self.device)
     
    def predict_from_data_iterator(self,
                               data_iterator,
                               save_probabilities: bool = False,
                               num_processes_segmentation_export: int = default_num_processes):
        
        
        """
        Each element from data_iterator must return a dict with keys:
        'data', 'ofile', and 'data_properties'. We return segmentation predictions,
        and now also save classification logits.
        """
        classification_outputs = {}
        
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # Convert input to FP16 if on CUDA
                # if self.device.type == 'cuda':
                #     data = data.half()

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # Segmentation output
                with torch.cuda.amp.autocast():
                    prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                # Classification output using encoder features
                    with torch.no_grad():
                        enc_feats = self.network.encoder(data.to(self.device))
                        # class_logits = self.network.ClassificationHead(enc_feats[-1].unsqueeze(0))
                        enc_feats = [enc.unsqueeze(0) for enc in enc_feats]
                        class_logits = self.network.ClassificationHead(enc_feats)
                        class_logits = torch.softmax(class_logits, dim=1).squeeze().tolist()

                # Store classification output (by basename of output file)
                case_id = os.path.basename(ofile) if ofile else f"case_{len(classification_outputs)}"
                classification_outputs[case_id] = class_logits

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                            self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                self.configuration_manager, self.label_manager,
                                properties,
                                save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'Done with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        # Save classification logits to JSON
        output_dir = os.path.dirname(ofile) if ofile else os.getcwd()
        maybe_mkdir_p(output_dir)
        with open(join(output_dir, "classification_logits.json"), "w") as f:
            json.dump(classification_outputs, f, indent=2)

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

        return ret

# Patch a static version of build_network_architecture into nnUNetTrainerWithClassification
@staticmethod
def static_build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                      num_input_channels, num_output_channels, enable_deep_supervision=True):
    network = nnUNetTrainer.build_network_architecture(
        architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
        num_input_channels, num_output_channels, enable_deep_supervision
    )
    # Store encoder output channels from network for classification head for version 1
    # encoder_output_channels = network.encoder.output_channels
    # network.ClassificationHead = ClassificationHead(encoder_output_channels[-1], num_classes=3).to(torch.device("cuda"))
    # return network

    # Store encoder output channels from network for classification head for version 5
    encoder_output_channels = network.encoder.output_channels
    network.ClassificationHead = ClassificationHead(
        encoder_output_channels,
        num_classes=3
    ).to(torch.device('cuda', 1))
    return network

nnUNetTrainerWithClassification.build_network_architecture = static_build_network_architecture


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best_combined.pth')
    args = parser.parse_args()

    # Initialize predictor
    predictor = PredictorWithClassification(
        tile_step_size=0.66,
        use_gaussian=True,
        use_mirroring='auto',
        perform_everything_on_device=True,
        device=torch.device('cuda', 1),
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # Load trained model
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=args.model_dir,
        use_folds=(args.fold,),
        checkpoint_name=args.checkpoint
    )

    # Benchmark inference
    torch.cuda.synchronize()
    start_time = time()

    # Run inference (will save segmentations + classification logits)
    predictor.predict_from_files(
        list_of_lists_or_source_folder=args.input_dir,
        output_folder_or_list_of_truncated_output_files=args.output_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2
    )

    torch.cuda.synchronize()
    end_time = time()

    # Print elapsed time
    total_time_minutes = (end_time - start_time) / 60
    print(f"\nTotal Inference Time: {total_time_minutes:.2f} minutes")
