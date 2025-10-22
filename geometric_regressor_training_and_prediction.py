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


class geometric_basic_regressor_3D(pl.LightningModule):
    def __init__(self, 
                 loss_func = "mse",  
                 learning_rate = 0.0001,
                 input_channels = 1,
                 output_channels = 1,
                 mlp_hidden = 256,
                 dropout_p = 0.0):
        
        super().__init__()

        self.save_hyperparameters()

        # lifting to SE3
        self.lifting_layer = gnn.GLiftingConvSE3(in_channels = input_channels,
                                                 out_channels = 16,
                                                 kernel_size = 5,
                                                 padding = 2)

        # Define the encoding layers
        self.encoder1 = geometric_conv_block(16, 16, 32, kernel_size = 5)
        self.maxpooler1 = GMaxSpatialPool3d_fixed(2)
        
        self.encoder2 = geometric_conv_block(32, 32, 64, kernel_size = 5)
        self.maxpooler2 = GMaxSpatialPool3d_fixed(2)
        
        self.encoder3 = geometric_conv_block(64, 64, 128, kernel_size = 5)
        self.maxpooler3 = GMaxSpatialPool3d_fixed(2)
        
        self.encoder4 = geometric_conv_block(128, 128, 256, kernel_size = 5)

    
        # Define the regression layers
        self.global_pool = gnn.GAvgGlobalPool()
        self.regressor = nn.Sequential(nn.Flatten(),                      
                                       nn.Linear(256, mlp_hidden),
                                       nn.LayerNorm(mlp_hidden),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
                                       nn.Linear(mlp_hidden, output_channels)
                                       )
        nn.init.uniform_(self.regressor[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.regressor[-1].bias)

        self.loss_func = loss_func.lower()
        if self.loss_func not in {"mse", "l1", "smoothl1"}:
            raise ValueError("Invalid loss type. Use 'mse', 'l1', or 'smoothl1'.")
        self.learning_rate = learning_rate

    def forward(self, x):
        # lifting SE3
        x, H = self.lifting_layer(x)
        
        # Encodor
        x, H = self.encoder1(x, H)
        x, H = self.maxpooler1(x, H)
        x, H = self.encoder2(x, H)
        x, H = self.maxpooler2(x, H)
        x, H = self.encoder3(x, H)
        x, H = self.maxpooler3(x, H)
        x, H = self.encoder4(x, H)
        
        # regressor
        features = self.global_pool(x, H)
        out  = self.regressor(features.float())     

        return out


    def training_step(self, batch, batch_idx):
        x, y = batch

        # corrects shape
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        y = y.float()

        preds = self(x)
        loss  = self.compute_loss(preds, y)
    
        #mae = torch.mean(torch.abs(preds - y))
        
        # MAPE (%)
        denom = y.abs().clamp_min(1e-8)
        mape  = 100.0 * torch.mean(torch.abs((preds - y) / denom))
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        self.log("train_mape", mape, prog_bar=True, on_epoch=False, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
    
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        y = y.float()

        preds = self(x)
        loss  = self.compute_loss(preds, y)

        #mae = torch.mean(torch.abs(preds - y))
        
        denom = y.abs().clamp_min(1e-8)
        mape  = 100.0 * torch.mean(torch.abs((preds - y) / denom))
        
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("valid_mape", mape, prog_bar=True, on_epoch=True, on_step=True)
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
        preds = preds.float()
        targets = targets.float()
        if self.loss_func == "mse":
            return nn.MSELoss()(preds, targets)
        elif self.loss_func == "l1":
            return nn.L1Loss()(preds, targets)
        elif self.loss_func == "smoothl1":
            return nn.SmoothL1Loss(beta=0.5)(preds, targets)
        else:
            raise ValueError("Invalid loss type.")
            
            
            
shape_measure = measure_volume(scaler = 20000, log_transform = True)        
            
            
            
            

print("##############################################")                   
print("gcnn regression (volume)")                    
print("##############################################")             
print()
            
model_save_name = "geometric_regressor"

# defines the input folder paths
current_data_folder = root_folder + "\\data" + "\\regression" 
current_train_folder = current_data_folder + "\\train_subvolumes"
current_valid_folder = current_data_folder + "\\valid_augmented_subvolumes"
current_train_image_folder = current_train_folder + "\\image"
current_train_intracristal_folder = current_train_folder + "\\intracristal"
current_valid_image_folder = current_valid_folder + "\\image"
current_valid_intracristal_folder = current_valid_folder + "\\intracristal"
train_names = np.array(all_file_names(current_train_image_folder))
valid_names = np.array(all_file_names(current_valid_image_folder))

# prepares the dataloaders for the training process
training_data_unet_torch = direct_measure_training_dataset(img_input_path = current_train_image_folder, 
                                                           mask_input_path_1 = current_train_intracristal_folder,
                                                           measurement_method = shape_measure)
validation_data_unet_torch = direct_measure_validation_dataset(img_input_path = current_valid_image_folder, 
                                                               mask_input_path_1 = current_valid_intracristal_folder,
                                                               measurement_method = shape_measure)
train_dataloader = DataLoader(training_data_unet_torch, batch_size = 2, shuffle = True)
valid_dataloader = DataLoader(validation_data_unet_torch, batch_size = 2, shuffle = True)

# defines the regression model
model = geometric_basic_regressor_3D(loss_func = "mse", 
                                     learning_rate = 1e-4,
                                     input_channels = 1,
                                     output_channels = 1,
                                     mlp_hidden = 256,
                                     dropout_p = 0.0)

                        
# Pytorch Lightning Callbacks
model_save_name_to_use = model_save_name
checkpoint_callback = ModelCheckpoint(monitor="valid_loss", save_top_k=1, mode="min",
                                      dirpath= root_folder + "\\model_checkpoints\\" + model_save_name_to_use + "\\",
                                      filename = model_save_name_to_use + "-{epoch:02d}-{valid_loss:.4f}")
early_stopping_callback = EarlyStopping(monitor="valid_loss", patience=5, mode="min", verbose=True)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
progress_bar = RichProgressBar(refresh_rate = 1, leave = True)      
geometric_regressor_trainer = pl.Trainer(max_epochs = 100, accelerator = "gpu", devices = [1],
                                         callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor, progress_bar],
                                         precision = 32)
                 
# starts the training
geometric_regressor_trainer.fit(model, train_dataloader, valid_dataloader)            
            
            
            
            
            
# defines the input folder paths
model_save_name = "geometric_regressor"

current_data_folder = root_folder + "\\data" + "\\regression" 
current_test_folder = current_data_folder + "\\test_augmented_subvolumes"

current_test_image_folder = current_test_folder + "\\image"
current_test_intracristal_folder = current_test_folder + "\\intracristal"

test_names = np.array(all_file_names(current_test_image_folder))
        
model_save_name_to_use = model_save_name 
        
current_model_path = root_folder + "\\model_checkpoints\\" + model_save_name_to_use + "\\"
        
current_model_name = all_file_names(current_model_path)[0]
        
setpath(current_model_path)
            
current_model = geometric_basic_regressor_3D.load_from_checkpoint(current_model_name,
                                                                  loss_func = "mse", 
                                                                  learning_rate = 1e-4,
                                                                  input_channels = 1,
                                                                  output_channels = 1,
                                                                  mlp_hidden = 256,
                                                                  dropout_p = 0.0)
       
current_model.eval()

prediction_collection, groundtruth_collection = test_set_regression(model = current_model,
                                                                    input_volume_path = current_test_image_folder,
                                                                    input_groundtruth_path = current_test_intracristal_folder,
                                                                    measurement_method = shape_measure,
                                                                    scaler = 20000,
                                                                    log_transform = True)            
            
            

plt.figure(figsize = (6,6))
plt.scatter(groundtruth_collection, 
            prediction_collection,
            alpha = 0.05,
            color = "red")
max_val = np.max((groundtruth_collection.max(), prediction_collection.max()))
plt.plot((0,max_val), (0,max_val))

plt.title("GCNN regressor", fontsize = 18)
plt.xlabel("groundtruth", fontsize = 18)
plt.ylabel("measured vol", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)            
            
            
            
            
            
            
            
print("##############################################")                   
print("cnn regression (volume)")                    
print("##############################################")             
print()
            
model_save_name = "regressor"

# defines the input folder paths
current_data_folder = root_folder + "\\data" + "\\regression" 
current_train_folder = current_data_folder + "\\train_subvolumes"
current_valid_folder = current_data_folder + "\\valid_augmented_subvolumes"
current_train_image_folder = current_train_folder + "\\image"
current_train_intracristal_folder = current_train_folder + "\\intracristal"
current_valid_image_folder = current_valid_folder + "\\image"
current_valid_intracristal_folder = current_valid_folder + "\\intracristal"
train_names = np.array(all_file_names(current_train_image_folder))
valid_names = np.array(all_file_names(current_valid_image_folder))

# prepares the dataloaders for the training process
training_data_unet_torch = direct_measure_training_dataset(img_input_path = current_train_image_folder, 
                                                           mask_input_path_1 = current_train_intracristal_folder,
                                                           measurement_method = shape_measure)
validation_data_unet_torch = direct_measure_validation_dataset(img_input_path = current_valid_image_folder, 
                                                               mask_input_path_1 = current_valid_intracristal_folder,
                                                               measurement_method = shape_measure)
train_dataloader = DataLoader(training_data_unet_torch, batch_size = 2, shuffle = True)
valid_dataloader = DataLoader(validation_data_unet_torch, batch_size = 2, shuffle = True)

# defines the regression model
model = regressor_3D(loss_func = "mse",  
                     learning_rate = 1e-4,
                     input_channels = 1,
                     output_channels = 1,
                     mlp_hidden = 256,
                     dropout_p = 0.0)
                        
# Pytorch Lightning Callbacks
model_save_name_to_use = model_save_name
checkpoint_callback = ModelCheckpoint(monitor="valid_loss", save_top_k=1, mode="min",
                                      dirpath= root_folder + "\\model_checkpoints\\" + model_save_name_to_use + "\\",
                                      filename = model_save_name_to_use + "-{epoch:02d}-{valid_loss:.4f}")
early_stopping_callback = EarlyStopping(monitor="valid_loss", patience=5, mode="min", verbose=True)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
progress_bar = RichProgressBar(refresh_rate = 1, leave = True)
#progress_bar = TQDMProgressBar(refresh_rate=1)        
regressor_trainer = pl.Trainer(max_epochs = 100, accelerator = "gpu", devices = [1],
                               callbacks = [checkpoint_callback, early_stopping_callback, lr_monitor, progress_bar],
                               precision = 32)
                 
# starts the training
regressor_trainer.fit(model, train_dataloader, valid_dataloader)            
            
            
            
            
            
            
            
            
# defines the input folder paths
model_save_name = "regressor"

current_data_folder = root_folder + "\\data" + "\\regression" 
current_test_folder = current_data_folder + "\\test_augmented_subvolumes"

current_test_image_folder = current_test_folder + "\\image"
current_test_intracristal_folder = current_test_folder + "\\intracristal"

test_names = np.array(all_file_names(current_test_image_folder))
        
model_save_name_to_use = model_save_name 
        
current_model_path = root_folder + "\\model_checkpoints\\" + model_save_name_to_use + "\\"
        
current_model_name = all_file_names(current_model_path)[0]
        
setpath(current_model_path)
            
current_model = regressor_3D.load_from_checkpoint(current_model_name,
                                                  loss_func = "mse", 
                                                  learning_rate = 1e-4,
                                                  input_channels = 1,
                                                  output_channels = 1,
                                                  mlp_hidden = 256,
                                                  dropout_p = 0.0)
       
current_model.eval()

prediction_collection, groundtruth_collection = test_set_regression(model = current_model,
                                                                    input_volume_path = current_test_image_folder,
                                                                    input_groundtruth_path = current_test_intracristal_folder,
                                                                    measurement_method = shape_measure,
                                                                    scaler = 20000,
                                                                    log_transform = True)            
            
            

plt.figure(figsize = (6,6))
plt.scatter(groundtruth_collection, 
            prediction_collection,
            alpha = 0.05,
            color = "red")
max_val = np.max((groundtruth_collection.max(), prediction_collection.max()))
plt.plot((0,max_val), (0,max_val))

plt.title("regular regressor", fontsize = 18)
plt.xlabel("groundtruth", fontsize = 18)
plt.ylabel("measured vol", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)                  
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


