# -*- coding: utf-8 -*-
"""
Various Convolutional Neural Networks for Image Segmentation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.nn.utils as utils
import lightning as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar, RichProgressBar
from torch.utils.data import DataLoader, Dataset
import keras
import sys
import math
import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.io as skio
import pickle
from volumetric_augmentations import *
warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')



###############################################################
###############################################################
# folder paths 
###############################################################
###############################################################
root_folder = os.getcwd()




###############################################################
###############################################################
# file management functions
###############################################################
###############################################################

def setpath(path):
    '''
    changes working directory, creates folder if path doesn't exist
    '''
    try:
        os.chdir(path)
    except OSError:
        os.makedirs(path)
        os.chdir(path)           


def all_file_names(folder, file_format = None):
    '''
    list all image filenames of a given format in a folder,
    if file_format == None, returns all files regardless of format.
    '''
    if file_format is None:
        names = os.listdir(folder)
    else:
        names = os.listdir(folder)
        for name in names:
            if name[-len(file_format):] != file_format:
                names.remove(name)
    return names


def visualize(img):
    plt.figure(figsize = (10, 10))
    plt.imshow(img, cmap='gray')






###############################################################
###############################################################
# Volume Measurement
###############################################################
###############################################################

def measure_volume(scaler = 1,
                   log_transform = False):
    
    assert scaler > 0
    def method(input_data):

        if log_transform == True:
            volume_in_voxels = np.log1p(np.sum(input_data)/scaler)
        else:
            volume_in_voxels = np.sum(input_data)/scaler
    
        return np.array([volume_in_voxels], dtype=np.float32)    
    
    return method # volume     




def reversion_to_original_val_range(pred, 
                                    scaler = 1,
                                    log_transform = False):
    
    
    
    
    if log_transform == True:
        reverted_val = np.expm1(pred) * scaler
    else:
        reverted_val = pred * scaler
    
    return reverted_val







class IoU_Loss_torch(nn.Module):
    
    
    def __init__(self, smooth = 1e-6):
        
        super().__init__()
        
        self.smooth = smooth



    def forward(self, preds, targets):
 
        preds = preds.view(preds.size(0), -1)
        
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim = 1)
        
        union = preds.sum(dim = 1) + targets.sum(dim = 1) - intersection

        
        iou = (intersection + self.smooth) / (union + self.smooth)

        
        return 1 - iou.mean()


###############################################################
###############################################################
# Image Regression Dataloaders and Generators
###############################################################
###############################################################


class direct_measure_training_dataset(Dataset):
    
    def __init__(self, img_input_path, 
                       mask_input_path_1,
                       measurement_method):
        
        """
        measurement_measurement(mask_volume) = (a,b,c,...) 
        """
        
        self.img_input_path = img_input_path
        self.mask_input_path_1 = mask_input_path_1


        # mask names are identical to image names, so no need to use both
        self.img_names = np.array(all_file_names(self.img_input_path))
        
        
        # augmentation parameters definition
        self.aug_pipeline = Augmenter([TransformSpec(random_flip3d, p=1.0),
                                       TransformSpec(random_affine3d, p=0.8, 
                                                     kwargs=dict(degrees=(-20,20),
                                                                 scales=(0.8,1.5),
                                                                 translate=((-12,12),(-12,12),(-12,12)),
                                                                 shears=((-4, 4), (-4, 4), (-4, 4)))),
                                       TransformSpec(random_elastic3d, p=0.3, 
                                                     kwargs=dict(sigma=(3,5), 
                                                                 alpha=(0.75,1.75)
                                                                 )),
                                       # random_intensity will be a no-op for masks via is_mask=True
                                       TransformSpec(random_intensity, p=0.67, 
                                                     kwargs=dict(gamma=(0.8,1.2), 
                                                                 brightness=(-0.1,0.1), 
                                                                 contrast=(0.9,1.1),
                                                                 gaussian_noise_std=(0.0, 0.03)
                                                                 ))
                                       ], seed=None)
        
        self.measurement_method = measurement_method




    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        # loads the image
        current_img_load_path = self.img_input_path + "\\" + self.img_names[idx]
        current_mask_load_path_1 = self.mask_input_path_1 + "\\" + self.img_names[idx]
        
        
        current_img_volume = np.load(current_img_load_path)
        current_mask_volume_1 = np.load(current_mask_load_path_1)
        
        
        # normalizes the image between 0 and 1
        current_img_volume = current_img_volume/255



        # augments the datapoint
        (image_subvol_aug, 
         mito_subvol_1_aug) = self.aug_pipeline.apply_all([current_img_volume, 
                                                           current_mask_volume_1], 
                                                           mask_flags=[False, 
                                                                       True])
        image_subvol_aug = image_subvol_aug.astype(np.float32)
        mito_subvol_1_aug = mito_subvol_1_aug.astype(np.float32)
        mask_combined = mito_subvol_1_aug
            
        
        # measures shape parameters
        target_params = self.measurement_method(mask_combined[0])
        target_params = np.array(target_params, 
                                 dtype=np.float32)

        
        # make the image anisotropic
        anisotropic_image_subvol_aug_padded = image_subvol_aug
        
        
        
        # returns input image and output params
        processed_input_array = torch.from_numpy(anisotropic_image_subvol_aug_padded)
        processed_output_array = torch.from_numpy(target_params)
        
        
        return processed_input_array, processed_output_array





class direct_measure_validation_dataset(Dataset):
    
    def __init__(self, img_input_path, 
                       mask_input_path_1,
                       measurement_method):
        
        self.img_input_path = img_input_path
        self.mask_input_path_1 = mask_input_path_1

        # mask names are identical to image names, so no need to use both
        self.img_names = np.array(all_file_names(self.img_input_path))

        self.measurement_method = measurement_method



    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):

        # loads the image
        current_img_load_path = self.img_input_path + "\\" + self.img_names[idx]
        current_mask_load_path_1 = self.mask_input_path_1 + "\\" + self.img_names[idx]
        
        
        current_img_volume = np.load(current_img_load_path)
        current_mask_volume_1 = np.load(current_mask_load_path_1)
        
        
        image_subvol_aug = current_img_volume.astype(np.float32)
        mito_subvol_1_aug = current_mask_volume_1.astype(np.float32)
        mask_combined = mito_subvol_1_aug


        # measures shape parameters
        target_params = self.measurement_method(mask_combined[0])
        target_params = np.array(target_params, 
                                 dtype=np.float32)

        
        # make the image anisotropic
        anisotropic_image_subvol_aug_padded = image_subvol_aug
                                                                      
                                                                      
        # returns input image and output params
        processed_input_array = torch.from_numpy(anisotropic_image_subvol_aug_padded)
        processed_output_array = torch.from_numpy(target_params)
        
        
        return processed_input_array, processed_output_array




class segmetation_3D_unet_training_dataset(Dataset):
    
    def __init__(self, img_input_path, 
                       mask_input_path_1):
        
        self.img_input_path = img_input_path
        self.mask_input_path_1 = mask_input_path_1


        # mask names are identical to image names, so no need to use both
        self.img_names = np.array(all_file_names(self.img_input_path))
        
        # augmentation parameters definition
        self.aug_pipeline = Augmenter([TransformSpec(random_flip3d, p=1.0),
                                       TransformSpec(random_affine3d, p=0.8, 
                                                     kwargs=dict(degrees=(-20,20),
                                                                 scales=(0.9,1.1),
                                                                 translate=((-12,12),(-12,12),(-12,12)),
                                                                 shears=((-4, 4), (-4, 4), (-4, 4)))),
                                       TransformSpec(random_elastic3d, p=0.3, 
                                                     kwargs=dict(sigma=(3,5), 
                                                                 alpha=(0.75,1.75)
                                                                 )),
                                       # random_intensity will be a no-op for masks via is_mask=True
                                       TransformSpec(random_intensity, p=0.67, 
                                                     kwargs=dict(gamma=(0.8,1.2), 
                                                                 brightness=(-0.1,0.1), 
                                                                 contrast=(0.9,1.1),
                                                                 gaussian_noise_std=(0.0, 0.03)
                                                                 ))
                                       ], seed=None)


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        # loads the image
        current_img_load_path = self.img_input_path + "\\" + self.img_names[idx]
        current_mask_load_path_1 = self.mask_input_path_1 + "\\" + self.img_names[idx]
        
        
        current_img_volume = np.load(current_img_load_path)
        current_mask_volume_1 = np.load(current_mask_load_path_1)
        


        # normalizes the image between 0 and 1
        current_img_volume = current_img_volume/255


        # augments the datapoint
        (image_subvol_aug, 
         mito_subvol_1_aug) = self.aug_pipeline.apply_all([current_img_volume, 
                                                           current_mask_volume_1], 
                                                           mask_flags=[False, 
                                                                       True])
        

                                                           
        image_subvol_aug = image_subvol_aug.astype(np.float32)
        mito_subvol_1_aug = mito_subvol_1_aug.astype(np.float32)

            
        mask_combined = mito_subvol_1_aug
       


        # returns input image and input mask as a single input with 2 channels + the output mask
        processed_input_array = torch.from_numpy(image_subvol_aug)
        processed_output_array = torch.from_numpy(mask_combined)
        
        
        return processed_input_array, processed_output_array





class segmetation_3D_unet_validation_dataset(Dataset):
    
    def __init__(self, img_input_path, 
                       mask_input_path_1):
        
        self.img_input_path = img_input_path
        self.mask_input_path_1 = mask_input_path_1


        # mask names are identical to image names, so no need to use both
        self.img_names = np.array(all_file_names(self.img_input_path))


    def __len__(self):
        return len(self.img_names)


    def __getitem__(self, idx):
        
        # loads the image
        current_img_load_path = self.img_input_path + "\\" + self.img_names[idx]
        current_mask_load_path_1 = self.mask_input_path_1 + "\\" + self.img_names[idx]
        
        
        current_img_volume = np.load(current_img_load_path)
        current_mask_volume_1 = np.load(current_mask_load_path_1)
        
        
        image_subvol_aug = current_img_volume.astype(np.float32)
        mito_subvol_1_aug = current_mask_volume_1.astype(np.float32)

        

        mask_combined = mito_subvol_1_aug


        # returns input image and input mask as a single input with 2 channels + the output mask
        processed_input_array = torch.from_numpy(image_subvol_aug)
        processed_output_array = torch.from_numpy(mask_combined)
        
        
        return processed_input_array, processed_output_array








###############################################################
###############################################################
# prediction
###############################################################
###############################################################

def test_set_regression(model,
                        input_volume_path,
                        input_groundtruth_path,
                        measurement_method,
                        scaler,
                        log_transform = False):
    

    input_names = np.array(all_file_names(input_volume_path))
    
    prediction_collection = []
    groundtruth_collection = []
    
    
    with torch.no_grad():
        device = next(model.parameters()).device
        for i in tqdm(np.arange(len(input_names))):
            current_input_name = input_names[i]
    
            current_img_load_path = input_volume_path + "\\" + current_input_name
            current_groundtruth_load_path = input_groundtruth_path + "\\" + current_input_name
        
        
            current_input_volume = np.load(current_img_load_path)
            current_input_groundtruth_mask = np.load(current_groundtruth_load_path)
            
            current_input_volume = current_input_volume.astype(np.float32)
            current_input_groundtruth_mask = current_input_groundtruth_mask.astype(np.float32)
            
            # measures groundtruth shape param
            current_groundtruth_params = measurement_method(current_input_groundtruth_mask[0])
            current_groundtruth_params = np.array(current_groundtruth_params, 
                                                  dtype=np.float32)
        
        
            # make the image anisotropic
            anisotropic_current_input_volume_padded = current_input_volume
                                                                                                                
                                                                                                                

            # formats the shape correctly, so its batch size first, in this case 1:
            anisotropic_current_input_volume_padded = anisotropic_current_input_volume_padded[np.newaxis]
            current_groundtruth_params = current_groundtruth_params[np.newaxis]
            
            
            # returns input image and output params
            processed_input_array = torch.from_numpy(anisotropic_current_input_volume_padded)

    
        
            # formatting and preparation
            processed_input_array = processed_input_array.to(device=device)
            processed_input_array = processed_input_array.contiguous(memory_format=torch.channels_last_3d)

        
            # performs prediction
            current_prediction = model(processed_input_array).cpu().numpy()
            current_prediction = current_prediction[0]
    
    
            reverted_prediction = reversion_to_original_val_range(current_prediction, 
                                                                 scaler = scaler,
                                                                 log_transform = log_transform)
    
            reverted_groundtruth = reversion_to_original_val_range(current_groundtruth_params[0], 
                                                                   scaler = scaler,
                                                                   log_transform = log_transform)
    
    
            # collects the results
            prediction_collection.append(reverted_prediction)
            groundtruth_collection.append(reverted_groundtruth)
            
            
    prediction_collection = np.array(prediction_collection)
    groundtruth_collection = np.array(groundtruth_collection)
    
    
    return prediction_collection, groundtruth_collection






def test_set_unet_segment(model,
                          input_volume_path,
                          input_groundtruth_path,
                          prediction_folder_path = root_folder + "\\results"  + "\\segmentation"):

    
    input_names = np.array(all_file_names(input_volume_path))
    

    with torch.no_grad():
        device = next(model.parameters()).device
        for i in tqdm(np.arange(len(input_names))):
            current_input_name = input_names[i]
    
            current_img_load_path = input_volume_path + "\\" + current_input_name
            current_groundtruth_load_path = input_groundtruth_path + "\\" + current_input_name
        
        
            current_input_volume = np.load(current_img_load_path)
            current_input_groundtruth_mask = np.load(current_groundtruth_load_path)
            
            current_input_volume = current_input_volume.astype(np.float32)
            current_input_groundtruth_mask = current_input_groundtruth_mask.astype(np.float32)
            


            # prepares output
            input_array = current_input_volume
            
            input_array = input_array[np.newaxis]


         
       
            # returns input 
            processed_input_array = torch.from_numpy(input_array)

            
            
        
            # formatting and preparation
            processed_input_array = processed_input_array.to(device=device)
            processed_input_array = processed_input_array.contiguous(memory_format=torch.channels_last_3d)


        
            # performs prediction
            current_prediction = model(processed_input_array).cpu().numpy()
            current_prediction = current_prediction[0]
    

            # save results
            setpath(prediction_folder_path)
            np.save(current_input_name, current_prediction)
    










#############################################################################
# Regular Image Regressor        

class regressor_3D(pl.LightningModule):
    def __init__(self, 
                 loss_func = "mse", 
                 metric_func = 'accuracy', 
                 learning_rate = 0.0001,
                 input_channels = 1,
                 output_channels = 1,
                 mlp_hidden = 256,
                 dropout_p = 0.0):
        
        super().__init__()

        self.save_hyperparameters()

        # encoding layers
        self.encoder1 = self.conv_block(input_channels, 16, 32, kernel_size = 3)
        self.maxpooler1 = nn.MaxPool3d(2)
        
        self.encoder2 = self.conv_block(32, 32, 64, kernel_size = 3)
        self.maxpooler2 = nn.MaxPool3d(2)
        
        self.encoder3 = self.conv_block(64, 64, 128, kernel_size = 3)
        self.maxpooler3 = nn.MaxPool3d(2)
        
        self.encoder4 = self.conv_block(128, 128, 256, kernel_size = 3)

        
        # regression layers
        self.gap = nn.AdaptiveAvgPool3d(1)  
        self.regressor = nn.Sequential(nn.Flatten(),                      
                                       nn.Linear(256, mlp_hidden),
                                       nn.LayerNorm(mlp_hidden),
                                       nn.ReLU(inplace = True),
                                       nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
                                       nn.Linear(mlp_hidden, output_channels))


        nn.init.uniform_(self.regressor[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.regressor[-1].bias)


        self.loss_func = loss_func.lower()
        
        if self.loss_func not in {"mse", "l1", "smoothl1"}:
            raise ValueError("Invalid loss type. Use 'mse', 'l1', or 'smoothl1'.")
            
        self.learning_rate = learning_rate



    
    def conv_block(self, in_channels, mid_channels, out_channels, kernel_size = 3):
        
        padding = (kernel_size - 1) // 2

        conv1_internal = nn.Conv3d(in_channels, 
                                   mid_channels, 
                                   kernel_size = kernel_size, 
                                   padding = padding)
        
        # HE normal initialization
        init.kaiming_normal_(conv1_internal.weight, mode = 'fan_in', nonlinearity = 'relu')
        if conv1_internal.bias is not None:
            init.zeros_(conv1_internal.bias)



        conv2_internal = nn.Conv3d(mid_channels, 
                                   out_channels, 
                                   kernel_size = kernel_size, 
                                   padding = padding)
        
        # HE normal initialization
        init.kaiming_normal_(conv2_internal.weight, mode = 'fan_in', nonlinearity = 'relu')
        if conv2_internal.bias is not None:
            init.zeros_(conv2_internal.bias)


        return nn.Sequential(conv1_internal,
                             nn.BatchNorm3d(mid_channels),
                             nn.ReLU(inplace = True),
                             conv2_internal,
                             nn.BatchNorm3d(out_channels),
                             nn.ReLU(inplace = True))



    def forward(self, x):
        # Encodor
        enc1 = self.encoder1(x)
        mpl1 = self.maxpooler1(enc1)
        
        enc2 = self.encoder2(mpl1)
        mpl2 = self.maxpooler2(enc2)
        
        enc3 = self.encoder3(mpl2)
        mpl3 = self.maxpooler3(enc3)
        
        enc4 = self.encoder4(mpl3)
        
        
        # regressor 
        features = self.gap(enc4)           
        out  = self.regressor(features.float())     


        return out



    def training_step(self, batch, batch_idx):
        
        x, y = batch
        x = x.contiguous(memory_format = torch.channels_last_3d)

        # corrects shape
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        y = y.float()

        
        preds = self(x)
        loss  = self.compute_loss(preds, y)
    

        denom = y.abs().clamp_min(1e-8)
        mape  = 100.0 * torch.mean(torch.abs((preds - y) / denom))
        
        
        self.log("train_loss", loss, prog_bar = True, on_epoch = False, on_step = True)
        self.log("train_mape", mape, prog_bar = True, on_epoch = False, on_step = True)
        
        return loss





    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.contiguous(memory_format = torch.channels_last_3d)

        if y.ndim == 1:
            y = y.unsqueeze(-1)
        y = y.float()

        preds = self(x)
        loss  = self.compute_loss(preds, y)


        denom = y.abs().clamp_min(1e-8)
        mape  = 100.0 * torch.mean(torch.abs((preds - y) / denom))
        
        
        self.log("valid_loss", loss, prog_bar = True, on_epoch = True, on_step = True)
        self.log("valid_mape", mape, prog_bar = True, on_epoch = True, on_step = True)
        
        return loss




    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode = 'min', 
                                                         patience = 3, 
                                                         factor = 0.2
                                                         )
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, 
                                 "monitor": "valid_loss"}}




    def compute_loss(self, preds, targets):
        
        preds = preds.float()
        targets = targets.float()
        
        if self.loss_func == "mse":
            return nn.MSELoss()(preds, targets)
        
        elif self.loss_func == "l1":
            return nn.L1Loss()(preds, targets)
        
        elif self.loss_func == "smoothl1":
            return nn.SmoothL1Loss(beta = 0.5)(preds, targets)
        
        else:
            raise ValueError("Invalid loss type.")



    def on_fit_start(self):
        self.to(memory_format = torch.channels_last_3d)
        
        
        
        

#############################################################################
# Regular UNet  

class UNet3D(pl.LightningModule):
    def __init__(self, 
                 loss_func = "iou", 
                 metric_func = 'accuracy', 
                 learning_rate = 0.0001,
                 input_channels = 2,
                 first_kernel_size = (3, 3, 3)):
        
        super().__init__()
        self.save_hyperparameters()
        
        self.first_kernel_size = first_kernel_size
        self.loss_func = loss_func
        self.metric_func = metric_func
        self.learning_rate = learning_rate
        
        
        # encoder
        self.encoder1 = self.conv_block(input_channels, 32, 64, 
                                        kernel_size = self.first_kernel_size)
        self.maxpooler1 = nn.MaxPool3d(2)
        
        self.encoder2 = self.conv_block(64, 64, 128, kernel_size = 3)
        self.maxpooler2 = nn.MaxPool3d(2)
        
        self.encoder3 = self.conv_block(128, 128, 256, kernel_size = 3)
        self.maxpooler3 = nn.MaxPool3d(2)
        
        self.encoder4 = self.conv_block(256, 256, 512, kernel_size = 3)
        self.upconv1 = self.upconv_block(512, 512)
        
        
        # decoder
        self.decoder1 = self.conv_block(512 + 256, 256, 256, kernel_size = 3)
        self.upconv2 = self.upconv_block(256, 256)
            
            
        self.decoder2 = self.conv_block(256 + 128, 128, 128, kernel_size = 3)
        self.upconv3 = self.upconv_block(128, 128)
            
            
        self.decoder3 = self.conv_block(128+64, 64, 64, kernel_size = 3)

        

        # segmentation classification
        self.final_conv = nn.Conv3d(64, 1, kernel_size = 1)
        
        if self.loss_func.lower() == "mse":
            # linear output for regression
            self.final_conv_activation = nn.Identity()
            
        else:
            # sigmoid output for segmentation
            self.final_conv_activation = nn.Sigmoid()
            
        
        # applies HE normal initialization
        init.kaiming_normal_(self.final_conv.weight, mode = 'fan_in', nonlinearity = 'relu')
        if self.final_conv.bias is not None:
            init.zeros_(self.final_conv.bias)
        
    
        

    def conv_block(self, in_channels, mid_channels, out_channels, kernel_size = 3, dilation = 1):
        
        # normalize to tuples
        if isinstance(kernel_size, int): 
            kernel_size = (kernel_size,) * 3
        if isinstance(dilation, int): 
            dilation = (dilation,) * 3

        # padding deal with
        padding_1 = tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation))
        padding_2 = tuple(((k - 1) * 1) // 2 for k in kernel_size)


        conv1_internal = nn.Conv3d(in_channels, 
                                   mid_channels, 
                                   kernel_size = kernel_size, 
                                   dilation = dilation,
                                   padding = padding_1)
        
        # HE normal initialization
        init.kaiming_normal_(conv1_internal.weight, mode = 'fan_in', nonlinearity = 'relu')
        if conv1_internal.bias is not None:
            init.zeros_(conv1_internal.bias)



        conv2_internal = nn.Conv3d(mid_channels, 
                                   out_channels, 
                                   kernel_size = kernel_size, 
                                   padding = padding_2)
        
        # HE normal initialization
        init.kaiming_normal_(conv2_internal.weight, mode = 'fan_in', nonlinearity = 'relu')
        if conv2_internal.bias is not None:
            init.zeros_(conv2_internal.bias)


        return nn.Sequential(conv1_internal,
                             nn.BatchNorm3d(mid_channels),
                             nn.ReLU(inplace = True),
                             conv2_internal,
                             nn.BatchNorm3d(out_channels),
                             nn.ReLU(inplace = True))



    
    def upconv_block(self, in_channels, out_channels):
        
        
        upconv_internal = nn.ConvTranspose3d(in_channels, 
                                             out_channels, 
                                             kernel_size = 2, 
                                             stride = 2)
        relu_internal = nn.ReLU(inplace = True)
            
        
        # HE normal initialization
        init.kaiming_normal_(upconv_internal.weight, mode = 'fan_in', nonlinearity = 'relu')
        if upconv_internal.bias is not None:
            init.zeros_(upconv_internal.bias)

        return nn.Sequential(upconv_internal,
                             relu_internal)





    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        mpl1 = self.maxpooler1(enc1)
        
        enc2 = self.encoder2(mpl1)
        mpl2 = self.maxpooler2(enc2)
        
        enc3 = self.encoder3(mpl2)
        mpl3 = self.maxpooler3(enc3)
        
        enc4 = self.encoder4(mpl3)
        
        
        
        # Decoding path
        upc1 = self.upconv1(enc4)
        conc1 = torch.cat([enc3, upc1], dim = 1)
        dec1 = self.decoder1(conc1)
        
        upc2 = self.upconv2(dec1)
        conc2 = torch.cat([enc2, upc2], dim = 1)
        dec2 = self.decoder2(conc2)
        
        upc3 = self.upconv3(dec2)
        conc3 = torch.cat([enc1, upc3], dim = 1)
        dec3 = self.decoder3(conc3)
        
        
        # segmentation prediction path    
        out = self.final_conv(dec3)
        out = self.final_conv_activation(out)
    

        return out



    def training_step(self, batch, batch_idx):
        
        x, y = batch
        x = x.contiguous(memory_format = torch.channels_last_3d)
        y = y.contiguous(memory_format = torch.channels_last_3d)
        
        preds = self(x)
        loss = self.compute_loss(preds, y)
        
        acc = self.accuracy(preds, y)
        
        self.log("train_loss", loss, prog_bar = True, on_epoch = False, on_step = True)
        self.log("train_acc", acc, prog_bar = True, on_epoch = False, on_step = True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.contiguous(memory_format = torch.channels_last_3d)
        y = y.contiguous(memory_format = torch.channels_last_3d)
        
        preds = self(x)
        loss = self.compute_loss(preds, y)
        
        acc = self.accuracy(preds, y)
        
        self.log("valid_loss", loss, prog_bar = True, on_epoch = True, on_step = True)
        self.log("valid_acc", acc, prog_bar = True, on_epoch = True, on_step = True)
        
        return loss



    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode = 'min', 
                                                         patience = 3, 
                                                         factor = 0.2)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, 
                                 "monitor": "valid_loss"}}




    def compute_loss(self, preds, targets):
        
        if self.loss_func == "f1":
            loss = F1_Loss_torch()(preds, targets)
            
        elif self.loss_func == "iou":
            loss = IoU_Loss_torch()(preds, targets)
            
        elif self.loss_func == "mse":
            loss = nn.MSELoss()(preds, targets)
            
        else:
            raise ValueError("Invalid loss type.")
            
        return loss



    def accuracy(self, preds, targets):
        
        
        preds = preds > 0.5
        targets = targets > 0.5
        
        
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

   
        
        correct_per_image = (preds == targets).sum(dim = 1).float()  
        total_per_image = preds.size(1)  

        
        accuracy_per_image = correct_per_image / total_per_image
        
        
        return accuracy_per_image.mean()



    def on_fit_start(self):
        self.to(memory_format = torch.channels_last_3d)
