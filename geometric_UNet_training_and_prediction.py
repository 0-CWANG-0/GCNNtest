import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import lightning as pl
from torch import optim
import gconv.gnn as gnn
from torch import Tensor
import numpy as np

from support import * 





class GMaxSpatialPool3d_fixed(nn.MaxPool3d):
    """
    Performs spatial max pooling on 3d spatial inputs.
    """

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
               
        y = super().forward(x.flatten(1, 2))
        D2, H2, W2 = y.shape[-3:]
        
        return y.view(*x.shape[:3], D2, H2, W2), H
    
    
    
    


class geometric_conv_block(nn.Module):
    def __init__(self, in_channels, 
                       mid_channels, 
                       out_channels, 
                       kernel_size = 5,
                       stride= 1):
        
        super().__init__()
        padding = (kernel_size - 1) // 2 - 1
        
        self.gconv1_internal = gnn.GSeparableConvSE3(in_channels, 
                                                     mid_channels, 
                                                     kernel_size = kernel_size,
                                                     padding = padding,
                                                     stride=1)
        self.normalization_1 = gnn.GBatchNorm3d(mid_channels)
        self.activation_1 = nn.ReLU(inplace=True)
        
        self.gconv2_internal = gnn.GSeparableConvSE3(mid_channels, 
                                                     out_channels, 
                                                     kernel_size = kernel_size,
                                                     padding = padding,
                                                     stride=stride)
        self.normalization_2 = gnn.GBatchNorm3d(out_channels)
        self.activation_2 = nn.ReLU(inplace=True)
        

    def forward(self, x, H):
        
        x, H = self.gconv1_internal(x, H)  
        x, H = self.normalization_1(x, H)         
        x = self.activation_1(x)
        
        x, H = self.gconv2_internal(x, H)
        x, H = self.normalization_2(x, H)
        x = self.activation_2(x)
        
        return x, H




class geometric_upconv_block(nn.Module):


    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = 3,
                 scale_factor = (2, 2, 2),
                 align_corners = False):
        
        
        super().__init__()

        

        padding = (kernel_size - 1) // 2 - 1

        
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        

        self.gconv_internal = gnn.GSeparableConvSE3(in_channels,
                                                    out_channels,
                                                    kernel_size = kernel_size,
                                                    padding = padding)
        
        self.normalization_internal = gnn.GBatchNorm3d(out_channels)
        
        self.relu_internal = nn.ReLU(inplace = True)
        



    def forward(self, x: torch.Tensor, H: torch.Tensor):
        # x: (B, C, R, D, H, W)
        B, C, R, D, Hs, Ws = x.shape

        # Merge (C, R)
        x_cr = x.flatten(1, 2)  # (B, C*R, D, H, W)
        x_cr = F.interpolate(x_cr,
                             scale_factor = self.scale_factor,
                             mode = 'trilinear',
                             align_corners = self.align_corners)

        # Restore group dim R
        D2, H2, W2 = x_cr.shape[-3:]
        x_up = x_cr.contiguous().view(B, C, R, D2, H2, W2)

        # Equivariant conv + norm + act
        x_up, H = self.gconv_internal(x_up, H)
        x_up, H = self.normalization_internal(x_up, H)
        x_up = self.relu_internal(x_up)
        
        return x_up, H





class Geometric_UNet3D(pl.LightningModule):
    def __init__(self, 
                 learning_rate=0.0001,
                 input_channels = 2,
                 first_kernel_size = 5):
        
        super().__init__()
        self.save_hyperparameters()
        
        
        self.first_kernel_size = first_kernel_size
        self.learning_rate = learning_rate
        
        
        # initial lifting
        self.padding_first = (self.first_kernel_size - 1) // 2
        
        
        self.lifting_layer = gnn.GLiftingConvSE3(in_channels = input_channels,
                                                 out_channels = 16,
                                                 kernel_size = self.first_kernel_size,
                                                 padding = self.padding_first)
        
        # Define the 3D U-Net layers (encoder)
        self.encoder1 = geometric_conv_block(16, 16, 16, 
                                             kernel_size = self.first_kernel_size)
        self.maxpooler1 = GMaxSpatialPool3d_fixed(2)
        
        self.encoder2 = geometric_conv_block(16, 16, 32, 
                                             kernel_size = 5)
        self.maxpooler2 = GMaxSpatialPool3d_fixed(2)
        
        self.encoder3 = geometric_conv_block(32, 32, 32, 
                                             kernel_size = 5)
        self.maxpooler3 = GMaxSpatialPool3d_fixed(2)
        
        self.encoder4 = geometric_conv_block(32, 64, 64, 
                                             kernel_size = 5)
        
        self.upconv1 = geometric_upconv_block(64, 
                                              64,
                                              kernel_size = 5,
                                              scale_factor = 2,
                                              align_corners = False)

        
        # Define the 3D U-Net layers (decoder)
        self.decoder1 = geometric_conv_block(64 + 32, 32, 32, 
                                             kernel_size = 5)
        self.upconv2 = geometric_upconv_block(32, 
                                              32,
                                              kernel_size = 5,
                                              scale_factor = 2,
                                              align_corners = False)
            
            
        self.decoder2 = geometric_conv_block(32 + 32, 16, 16, 
                                             kernel_size = 5)
        self.upconv3 = geometric_upconv_block(16, 
                                              16,
                                              kernel_size = 5,
                                              scale_factor = 2,
                                              align_corners = False)
            
        self.decoder3 = geometric_conv_block(16+16, 16, 16, 
                                             kernel_size = 5)

        
        
        self.group_pool = gnn.GAvgGroupPool() 

        # segmentation classification
        self.final_conv = nn.Conv3d(16, 1, kernel_size = 1)

        # sigmoid output for segmentation
        self.final_conv_activation = nn.Sigmoid()
            
        
        # applies HE normal initialization
        init.kaiming_normal_(self.final_conv.weight, mode='fan_in', nonlinearity='relu')
        if self.final_conv.bias is not None:
            init.zeros_(self.final_conv.bias)
        
    



    def forward(self, x):

        # lifting layer
        x, H = self.lifting_layer(x)                 

        # Encoding path
        enc1, H = self.encoder1(x, H)               
        mpl1, H = self.maxpooler1(enc1, H)          
        
        enc2, H = self.encoder2(mpl1, H)             
        mpl2, H = self.maxpooler2(enc2, H)
        
        enc3, H = self.encoder3(mpl2, H)           
        mpl3, H = self.maxpooler3(enc3, H)
        
        enc4, H = self.encoder4(mpl3, H)           
        
        # Decoding path
        upc1, H = self.upconv1(enc4, H)              
        conc1 = torch.cat([enc3, upc1], dim=1)       
        dec1, H = self.decoder1(conc1, H)            
        
        upc2, H = self.upconv2(dec1, H)              
        conc2 = torch.cat([enc2, upc2], dim=1)       
        dec2, H = self.decoder2(conc2, H)            
        
        upc3, H = self.upconv3(dec2, H)            
        conc3 = torch.cat([enc1, upc3], dim=1)       
        dec3, H = self.decoder3(conc3, H)            

        # projection
        dec3_project = self.group_pool(dec3)          
        
        # segmentation prediction path   
        out = self.final_conv(dec3_project)           
        out = self.final_conv_activation(out)     
        
        return out




    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.contiguous()
        y = y.contiguous()
        
        preds = self(x)
        loss = self.compute_loss(preds, y)
        
        acc = self.accuracy(preds, y)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=False, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.contiguous()
        y = y.contiguous()
        
        preds = self(x)
        loss = self.compute_loss(preds, y)
        
        acc = self.accuracy(preds, y)
        
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("valid_acc", acc, prog_bar=True, on_epoch=True, on_step=True)
        return loss



    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         patience=3, 
                                                         factor=0.2
                                                         )
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"}}




    def compute_loss(self, preds, targets):

        loss = IoU_Loss_torch()(preds, targets)

        return loss



    def accuracy(self, preds, targets):
        
        # Convert predictions to binary (0 or 1)
        preds = preds > 0.5
        targets = targets > 0.5
        
        # Flatten the tensors to compute accuracy
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

   
        # Calculate accuracy per image in the batch
        correct_per_image = (preds == targets).sum(dim=1).float()  # Correct predictions per image
        total_per_image = preds.size(1)  # Number of elements per image

        # Compute accuracy for each image in the batch
        accuracy_per_image = correct_per_image / total_per_image
        
        
        return accuracy_per_image.mean()

            
            
            
print("##############################################")                   
print("gcnn UNet")                    
print("##############################################")             
print()
            
model_save_name = "geometric_unet"

# defines the input folder paths
current_data_folder = root_folder + "\\data" + "\\segmentation" 
current_train_folder = current_data_folder + "\\train_subvolumes"
current_valid_folder = current_data_folder + "\\valid_augmented_subvolumes"
current_train_image_folder = current_train_folder + "\\image"
current_train_intracristal_folder = current_train_folder + "\\inner_mito"
current_valid_image_folder = current_valid_folder + "\\image"
current_valid_intracristal_folder = current_valid_folder + "\\inner_mito"
train_names = np.array(all_file_names(current_train_image_folder))
valid_names = np.array(all_file_names(current_valid_image_folder))

# prepares the dataloaders for the training process
training_data_unet_torch = segmetation_3D_unet_training_dataset(img_input_path = current_train_image_folder, 
                                                                mask_input_path_1 = current_train_intracristal_folder)
validation_data_unet_torch = segmetation_3D_unet_validation_dataset(img_input_path = current_valid_image_folder, 
                                                                    mask_input_path_1 = current_valid_intracristal_folder)
train_dataloader = DataLoader(training_data_unet_torch, batch_size = 1, shuffle = True)
valid_dataloader = DataLoader(validation_data_unet_torch, batch_size = 1, shuffle = True)


# defines the regression model
model = Geometric_UNet3D(learning_rate=1e-4,
                         input_channels = 1,
                         first_kernel_size = 5)




                        
# Pytorch Lightning Callbacks
model_save_name_to_use = model_save_name
checkpoint_callback = ModelCheckpoint(monitor="valid_loss", save_top_k=1, mode="min",
                                      dirpath= root_folder + "\\model_checkpoints\\" + model_save_name_to_use + "\\",
                                      filename = model_save_name_to_use + "-{epoch:02d}-{valid_loss:.4f}")
early_stopping_callback = EarlyStopping(monitor="valid_loss", patience=5, mode="min", verbose=True)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
progress_bar = RichProgressBar(refresh_rate = 1, leave = True)      
geometric_unet_trainer = pl.Trainer(max_epochs = 100, accelerator = "gpu", devices = [1],
                                    callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor, progress_bar],
                                    precision = 32)
                 
# starts the training
geometric_unet_trainer.fit(model, train_dataloader, valid_dataloader)            
            
            
            
            
            
# defines the input folder paths
model_save_name = "geometric_unet"

current_data_folder = root_folder + "\\data" + "\\segmentation" 
current_test_folder = current_data_folder + "\\test_subvolumes"

current_test_image_folder = current_test_folder + "\\image"
current_test_intracristal_folder = current_test_folder + "\\inner_mito"

test_names = np.array(all_file_names(current_test_image_folder))
        
model_save_name_to_use = model_save_name 
        
current_model_path = root_folder + "\\model_checkpoints\\" + model_save_name_to_use + "\\"
        
current_model_name = all_file_names(current_model_path)[0]
        
setpath(current_model_path)
            
current_model = Geometric_UNet3D.load_from_checkpoint(current_model_name,
                                                      learning_rate=1e-4,
                                                      input_channels = 1,
                                                      first_kernel_size = 5)

current_model.eval()

            
test_set_unet_segment(model = current_model,
                      input_volume_path = current_test_image_folder,
                      input_groundtruth_path = current_test_intracristal_folder,
                      prediction_folder_path = root_folder + "\\results"  + "\\gcnn_segmentation")            
            
            

            
            
            
            
            
            
            
            
            
            
            


