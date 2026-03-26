import torch
import random
import numpy as np
from torch import optim
from model import Model
from train import train_fn
from val import val_fn
from test import test_fn
from config import *
from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
from datetime import datetime
# Required to do below or gives error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loading_utils import get_dataloaders
from monai.networks.nets import UNet
from early_stopping import EarlyStopping

class ModelTrainer():
    def __init__(self, model, loss_fn):
        
        # Obtains self.train_df and self.val_df
        super().__init__()
       
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.n_classes = n_classes

        self.model = model
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # Set up seeds
        self._set_seed()

        # Set the today's time
        now = datetime.now()
        self.timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
  
    def _set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Ensure reproducibility on GPU
        torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
        torch.backends.cudnn.benchmark = False  # Avoids non-deterministic behavior

    def save_weights(self, filename_to_save,  model, start_time, optimizer, epoch, val_iou):
        # Save weights
        print(filename_to_save)

        torch.save({
            'model_state_dict' : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            
            'model_config': {
                "num_epochs": self.num_epochs,
                "lr": self.lr,
                'time': time.time() - start_time,
                'final_lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1,
                'val_iou': val_iou,
                }
            }, os.path.join('model_checkpoints', filename_to_save))
        

        self.writer.add_text("Model Saving", f"Best model saved at epoch {epoch+1}", epoch)

    def train(self, save_train_preds=True):
        start_time = time.time()

        # initialize the model with params from paper
        if self.model is None:
            self.model = Model(in_channels=1, base_feat=14, depth=4, n_classes=self.n_classes)
           
                    
        # Create a unique filename for saving the model
        filename_to_save = "_".join([
            str(self.timestamp)
        ]) + ".pth"
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=20, min_lr=1e-5)
        early_stopper = EarlyStopping(patience=15, min_delta=0.001, mode='max')

        target_layer = None
        try:
            target_layer = self.model.dec_blocks[-1]
        except:
            # print(self.model.parameters)
            print( "The target layer was not defined.")

        # Tensorboard logging
        self.writer = SummaryWriter(log_dir=os.path.join('runs', filename_to_save.replace('.pth','')))

        # Prepare data loaders
        train_loader, val_loader, test_loader = get_dataloaders()

        # Training loop
        for epoch in range(self.num_epochs):
            print(f"Epoch: {epoch + 1}/{self.num_epochs}")

            # Train the model
            if epoch == self.num_epochs - 1 and save_train_preds:

                # Train
                epoch_start = time.time()
                (train_loss,
                    train_dice,
                    train_iou,
                    train_precision,
                    train_recall,
                    train_specificity,
                    train_accuracy,
                    train_inference_time_per_frame) = train_fn(
                        self.loss_fn,
                        train_loader,
                        self.model,
                        optimizer,
                        device=self.device,
                        save_preds=True
                    )
                train_epoch_time  = time.time() - epoch_start

                # Validation
                epoch_start = time.time()
                (val_loss_val,
                    val_dice,
                    val_iou,
                    val_precision,
                    val_recall,
                    val_specificity,
                    val_accuracy,
                    val_inference_time_per_frame) = val_fn(
                        self.loss_fn,
                        val_loader,
                        self.model,
                        target_layer,
                        device=self.device,
                        save_preds=True
                    )
                val_epoch_time  = time.time() - epoch_start


            else:

                # Train
                epoch_start = time.time()
                (train_loss,
                    train_dice,
                    train_iou,
                    train_precision,
                    train_recall,
                    train_specificity,
                    train_accuracy,
                    train_inference_time_per_frame) = train_fn(
                        self.loss_fn,
                        train_loader,
                        self.model,
                        optimizer,
                        device=self.device,
                        save_preds=False
                    )
                train_epoch_time  = time.time() - epoch_start


                # Validation 
                epoch_start = time.time()                 
                (val_loss_val,
                    val_dice,
                    val_iou,
                    val_precision,
                    val_recall,
                    val_specificity,
                    val_accuracy,
                    val_inference_time_per_frame) = val_fn(
                        self.loss_fn,
                        val_loader,
                        self.model,
                        target_layer,
                        device=self.device,
                        save_preds=False
                    )
                val_epoch_time  = time.time() - epoch_start


            scheduler.step(val_iou)
            print(f"Train and validation MIoU score: {train_iou:.5f}% and {val_iou:.5f}% ")
            print(f"Train and validation mean dice coeff: {train_dice:.5f}% and {val_dice:.5f}%")
            print(f'Train epoch time: {train_epoch_time}')
            print(f"Validation epoch time: {val_epoch_time}")
            

            # Log to tensorboard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Mean Dice Coeff/Train", train_dice, epoch)
            self.writer.add_scalar("MIoU/Train", train_iou, epoch)
            self.writer.add_scalar("Mean Precision/Train", train_precision, epoch)
            self.writer.add_scalar("Mean Recall/Train", train_recall, epoch)
            self.writer.add_scalar("Mean Specificity/Train", train_specificity, epoch)
            self.writer.add_scalar("Mean Accuracy/Train", train_accuracy, epoch)
            self.writer.add_scalar("Mean Inference Per Frame/Train", train_inference_time_per_frame, epoch)
            self.writer.add_scalar("Epoch Time/Train", train_epoch_time, epoch)

            self.writer.add_scalar("Loss/Validation", val_loss_val, epoch)
            self.writer.add_scalar("Mean Dice Coeff/Validation", val_dice, epoch)
            self.writer.add_scalar("MIoU/Validation", val_iou, epoch)
            self.writer.add_scalar("Mean Precision/Validation", val_precision, epoch)
            self.writer.add_scalar("Mean Recall/Validation", val_recall, epoch)
            self.writer.add_scalar("Mean Specificity/Validation", val_specificity, epoch)
            self.writer.add_scalar("Mean Accuracy/Validation", val_accuracy, epoch)
            self.writer.add_scalar("Mean Inference Per Frame/Validation", val_inference_time_per_frame, epoch)
            self.writer.add_scalar("Epoch Time/Validation", val_epoch_time, epoch)

            self.writer.add_scalar("Learning Rate/Epoch", optimizer.param_groups[0]['lr'], epoch + 1)

            # Evaluate if need to stop
            should_stop, is_best = early_stopper(val_iou)

            if is_best:
                # Save weights
                self.save_weights(filename_to_save, self.model, start_time, optimizer, epoch, val_iou)


            if should_stop:

                # Train
                (train_loss,
                    train_dice,
                    train_iou,
                    train_precision,
                    train_recall,
                    train_specificity,
                    train_accuracy,
                    train_inference_time_per_frame) = train_fn(
                        self.loss_fn,
                        train_loader,
                        self.model,
                        optimizer,
                        device=self.device,
                        save_preds=True
                    )

                # Validation
                (val_loss_val,
                    val_dice,
                    val_iou,
                    val_precision,
                    val_recall,
                    val_specificity,
                    val_accuracy,
                    val_inference_time_per_frame) = val_fn(
                        self.loss_fn,
                        val_loader,
                        self.model,
                        target_layer,
                        device=self.device,
                        save_preds=True
                    )

                print(f"Early stopping triggered at epoch {epoch+1}")

                self.writer.add_text("EarlyStopping", f"Triggered at epoch {epoch+1}", epoch)

                break
                        

        # Test the model after training
        print(f"Testing model after epoch {epoch+1}")
        epoch_start = time.time()
        (test_loss_val,
            test_dice,
            test_iou,
            test_precision,
            test_recall,
            test_specificity,
            test_accuracy,
            test_inference_time_per_frame) = test_fn(
                self.loss_fn,
                test_loader,
                self.model,
                target_layer,
                device=self.device,
                save_preds=True
            )
        test_epoch_time  = time.time() - epoch_start
        

        print(f"Test dice score: {test_dice:.5f}%")
        print(f"Test iou score: {test_iou:.5f}%")
        print(f"Test epoch time: {test_epoch_time}")

        # Log test metrics to tensorboard
        self.writer.add_scalar("Loss/Test", test_loss_val, epoch)
        self.writer.add_scalar("Mean Dice Coeff/Test", test_dice, epoch)
        self.writer.add_scalar("MIoU/Test", test_iou, epoch)
        self.writer.add_scalar("Mean Precision/Test", test_precision, epoch)
        self.writer.add_scalar("Mean Recall/Test", test_recall, epoch)
        self.writer.add_scalar("Mean Specificity/Test", test_specificity, epoch)
        self.writer.add_scalar("Mean Accuracy/Test", test_accuracy, epoch)
        self.writer.add_scalar("Mean Inference Per Frame/Test", test_inference_time_per_frame, epoch)
        self.writer.add_scalar("Epoch Time/Test", test_epoch_time, epoch)


        self.writer.close()
        print(f"Experiment {filename_to_save} completed.")


        # Save weights in case early stopping was not triggered
        self.save_weights(filename_to_save, self.model, start_time, optimizer, epoch, val_iou)

def main():

    loss_fn = DiceLoss(
        to_onehot_y=False,    # my y label is already BCZYX and does not need one hot conversion
        softmax=True,        # applies softmax to model logits
        include_background=False,  # optional: ignore background
        reduction='mean'
        )

    model = UNet(
        spatial_dims=3,      # 3D
        in_channels=1,        # grayscale CT input
        out_channels=n_classes, 
        channels=(16, 32, 64, 128),  # smaller channels for speed
        strides=(2, 2, 2),
        num_res_units=2,      # residual blocks per level
    ).to(device)

    # Set seed for reproducibility
    trainer = ModelTrainer(model=None, loss_fn=loss_fn)

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
