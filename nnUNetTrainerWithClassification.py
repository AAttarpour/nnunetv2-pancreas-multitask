"""
This file ingerits from nnUNetTrainer and adds a classification head to the nnUNet architecture.
so that we can train a nnUNet model for both segmentation and classification tasks.
The nnUNetTrainerWithClassification class builds upon the nnUNetTrainer class and modifies
the network architecture to include a classification head.
It also overrides the train, and validation step to compute both segmentation and classification losses.

"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.network_initialization import InitWeights_He
import torch
import torch.nn as nn
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
import numpy as np
from torch import distributed as dist
from typing import Tuple, Union, List
import os
import pandas as pd
from sklearn.metrics import f1_score
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from sklearn.utils.class_weight import compute_class_weight
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


# version 1: simple classification head; it operates on the features last layer of encoder
# class ClassificationHead(nn.Module):
#     """
#         A simple classification head for 3D data.
#         It uses an adaptive average pooling layer followed by a fully connected layer.
#     """
#     def __init__(self, in_channels, num_classes, dropout_p=0.3, hidden_dim=128):
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool3d(1)  # [B, C, 1, 1, 1]
#         self.flatten = nn.Flatten() # check if it keeps batch should be [B, C]
#         self.classifier = nn.Sequential(
#             nn.Linear(in_channels, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, x):
#         x = self.pool(x)
#         x = self.flatten(x)
#         return self.classifier(x)

# version 2: multi depth classification head; it operates on the features of the encoder
# class ClassificationHead(nn.Module):
#     def __init__(self, encoder_channels, num_classes, hidden_dim=128, dropout_p=0.3):
#         super().__init__()
#         self.global_pool = nn.AdaptiveMaxPool3d(1)
#         self.flatten = nn.Flatten()

#         total_channels = sum(encoder_channels)

#         self.classifier = nn.Sequential(
#             nn.Linear(total_channels, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(hidden_dim, num_classes)
#         )

#     def forward(self, encoder_features):
#         # Only use selected encoder features
#         pooled = [self.flatten(self.global_pool(f)) for f in encoder_features]
#         concat = torch.cat(pooled, dim=1)
#         out = self.classifier(concat)
#         return out

# version 3: multi depth classification head with feature adaptor; it operates on the features of the last three layers of the encoder
class ClassificationHead(nn.Module):
    def __init__(self, encoder_channels, num_classes=3, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        
        # Validate input
        if len(encoder_channels) < 3:
            raise ValueError("Encoder must have at least 3 feature maps")
        
        # Take the last 3 feature maps (now works with [256, 320, 320])
        selected_channels = encoder_channels[-3:]
        
        # Feature compression (adapts to actual channel dims)
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, 64, kernel_size=1, bias=False),  # Reduce to 64 channels
                nn.BatchNorm3d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((4, 4, 4))  # Downsample to (4,4,4)
            ) for ch in selected_channels
        ])
        
        # Enhanced fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(64 * 3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_features):
        # Input validation
        if len(encoder_features) < 3:
            raise ValueError(f"Expected ≥3 feature maps, got {len(encoder_features)}")
        
        # Process last 3 features
        adapted = []
        for i, adapter in enumerate(self.feature_adapters):
            feat = encoder_features[-(3 - i)]  # Get features from last to -3
            adapted.append(adapter(feat))
        
        # Concatenate and fuse
        x = torch.cat(adapted, dim=1)
        x = self.fusion(x).flatten(1)
        
        return self.classifier(x)

class nnUNetTrainerWithClassification(nnUNetTrainer):

    def initialize(self, *args, **kwargs):
        # Call the base class's initialize method first
        super().initialize(*args, **kwargs)

        # Load subtype labels CSV and store in self.subtype_dict
        subtype_file = os.path.join('nnUNet_preprocessed/Dataset003_PancreasMultiTask/subtype_results_train.csv')

        if not os.path.isfile(subtype_file):
            raise FileNotFoundError(f"Could not find subtype file at: {subtype_file}")

        df = pd.read_csv(subtype_file)
        self.subtype_dict = {
            row['Name'].replace(".nii.gz", ""): int(row['Subtype'])
            for _, row in df.iterrows()
        }

        # === Load subtype CSV and compute class weights ===
        # Uncomment the following lines if you want to compute class weights and use them in training

        # csv_path = "nnUNet_preprocessed/Dataset003_PancreasMultiTask/subtype_results_train.csv"
        # df = pd.read_csv(csv_path)
        # labels = df['Subtype'].values

        # # Compute and store class weights
        # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        # self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        # self.class_weights /= self.class_weights.sum()  # normalize if you want

        # self.print_to_log_file(f"Loaded class weights (normalized): {self.class_weights.tolist()}")
    

    # inherit from nnUNetTrainer, so same input arguments
    def build_network_architecture(self, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        network = super().build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        # Store encoder output channels from network for classification head for version 1
        # encoder_output_channels = network.encoder.output_channels
        # network.ClassificationHead = ClassificationHead(encoder_output_channels[-1], num_classes=3).to(self.device)
        # network.ClassificationHead.apply(InitWeights_He(1e-2))

        # Store encoder output channels from network for classification head for version 2
        # encoder_output_channels = network.encoder.output_channels
        # selected_encoder_channels = encoder_output_channels[:5]  # [32, 64, 128, 256, 320]
        # network.ClassificationHead = ClassificationHead(
        #     selected_encoder_channels,
        #     num_classes=3
        # ).to(self.device)
        # network.ClassificationHead.apply(InitWeights_He(1e-2))

        # Store encoder output channels from network for classification head for version 5
        encoder_output_channels = network.encoder.output_channels
        network.ClassificationHead = ClassificationHead(
            encoder_output_channels,
            num_classes=3
        ).to(self.device)
        return network
    
    # override configure_optimizers to use different learning rates for the classification head
    # def configure_optimizers(self):
    #     # Separate parameters explicitly
    #     base_params = []
    #     head_params = []
        
    #     for name, param in self.network.named_parameters():
    #         if 'ClassificationHead' in name:
    #             head_params.append(param)
    #         else:
    #             base_params.append(param)
        
    #     params = [
    #         {"params": base_params, 
    #         "lr": self.initial_lr,
    #         "weight_decay": self.weight_decay},
            
    #         {"params": head_params,
    #         "lr": self.initial_lr / 10,  # lower LR for classification
    #         "weight_decay": self.weight_decay}
    #     ]
        
    #     optimizer = torch.optim.SGD(
    #         params,
    #         momentum=0.99,
    #         nesterov=True
    #     )
        
    #     lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
    #     return optimizer, lr_scheduler

    # override train_step and validation_step to include classification loss
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        # get the subtype from the batch
        case_ids = batch['keys']
        subtype = torch.tensor(
            [self.subtype_dict[k] for k in case_ids],
            dtype=torch.long,
            device=self.device
        )

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if subtype is not None:
            subtype = subtype.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output = self.network(data)
            # Get the deepest feature map from the encoder
            # This is the output of the last encoder block before the bottleneck
            # This is where we will apply the classification head
            enc_features = self.network.encoder(data)

            # DEBUG PRINTING
            # for enc_feature in enc_features:
            #     print(f"Encoder features shape: {enc_feature.shape}")

            # for version 1
            # 32, 64, 128, 256, 320, 320 --> 3, 320, 4, 4, 6
            # class_logits = self.network.ClassificationHead(enc_features[-1])

            # for version 2
            # class_logits = self.network.ClassificationHead(enc_features[:5])

            # for version 3
            class_logits = self.network.ClassificationHead(enc_features)

            seg_loss = self.loss(seg_output, target)

            if subtype is not None:

                # uncomment this line if you want to use class weights
                #class_loss = nn.CrossEntropyLoss(weight=self.class_weights)(class_logits, subtype.long())

                class_loss = nn.CrossEntropyLoss()(class_logits, subtype.long())   
                total_loss = seg_loss + 0.3 * class_loss  # Weighted combo
            else:
                total_loss = seg_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    # same thing for validation step
    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']

        # get the subtype from the batch
        case_ids = batch['keys']
        subtype = torch.tensor(
            [self.subtype_dict[k] for k in case_ids],
            dtype=torch.long,
            device=self.device
        )

        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if subtype is not None:
            subtype = subtype.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            
            seg_output = self.network(data)
            # Get the deepest feature map from the encoder
            # This is the output of the last encoder block before the bottleneck
            # This is where we will apply the classification head
            enc_features = self.network.encoder(data)

            # for version 1
            # 32, 64, 128, 256, 320, 320 --> 3, 320, 4, 4, 6
            # class_logits = self.network.ClassificationHead(enc_features[-1])

            # for version 4
            # class_logits = self.network.ClassificationHead(enc_features[:5])

            # for version 5
            class_logits = self.network.ClassificationHead(enc_features)

            seg_loss = self.loss(seg_output, target)

            if subtype is not None:

                # uncomment this line if you want to use class weights
                # class_loss = nn.CrossEntropyLoss(weight=self.class_weights)(class_logits, subtype.long()) # 3x3 and subtype 3

                class_loss = nn.CrossEntropyLoss()(class_logits, subtype.long()) # 3x3 and subtype 3
                total_loss = seg_loss + 0.3 * class_loss # Weighted combo
                preds = torch.argmax(class_logits, dim=1)
                f1 = f1_score(subtype.cpu().numpy(), preds.cpu().numpy(), average='macro')
            else:
                total_loss = seg_loss
                f1 = 0

        result = {
            'loss': total_loss.detach().cpu().numpy(),
            'classification_f1': f1
        }

        # Keep original segmentation dice computation
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = target[0]

        axes = [0] + list(range(2, seg_output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:] if target.dtype != torch.bool else ~target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        result.update({
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        })

        return result
    
    # override on_validation_epoch_end to include classification accuracy
    # and mean fg dice
    # this is called at the end of each validation epoch
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            # Gather segmentation stats
            for name, var in zip(['tp', 'fp', 'fn'], [tp, fp, fn]):
                gathered = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, var)
                locals()[name] = np.vstack([i[None] for i in gathered]).sum(0)

            # Gather losses
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            # f1
            f1s = [None for _ in range(world_size)]
            dist.all_gather_object(f1s, outputs_collated['classification_f1'])
            f1 = np.mean(f1s)
        else:
            loss_here = np.mean(outputs_collated['loss'])

            # switch to f1
            f1 = np.mean(outputs_collated['classification_f1'])

        global_dc_per_class = [2 * i / (2 * i + j + k) if (2 * i + j + k) > 0 else 0 for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)

        # Store segmentation metrics
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

        # Store F1 values
        if 'classification_f1' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['classification_f1'] = []

        self.logger.my_fantastic_logging['classification_f1'].append(f1)


    def on_epoch_end(self):
        super().on_epoch_end()
        # Log classification F1
        if 'classification_f1' in self.logger.my_fantastic_logging:
            f1 = self.logger.my_fantastic_logging['classification_f1'][-1]
            self.print_to_log_file(f"Classification Macro F1: {np.round(f1, 4)}")
        else:
            f1 = 0.0

        # Get segmentation score
        dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        whole_pancreas_dsc = np.mean(dice_per_class[1:])  # label 1 and 2

        # Create a combined score metric (customizable)
        combined_score = (whole_pancreas_dsc + f1) / 2

        # Initialize if not yet done
        if not hasattr(self, '_best_combined_score'):
            self._best_combined_score = -np.inf

        # Save model only if improved
        if combined_score > self._best_combined_score and whole_pancreas_dsc > 0.7 and f1 > 0.6:
            self._best_combined_score = combined_score
            self.print_to_log_file(
                f"New best model: whole_dsc={whole_pancreas_dsc:.4f}, f1={f1:.4f}, combined={combined_score:.4f}"
            )
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_combined.pth'))
