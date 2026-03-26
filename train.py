import torch
from tqdm import tqdm
from config import *
import numpy as np
import time
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch.metrics as smp_metrics 
from model_training_utilities import ModelTrainingUtilities

save_predictions = ModelTrainingUtilities.save_predictions_as_imgs

def train_fn(loss_fn, loader, model, optimizer, device='cuda', save_preds=True, to_debug=False):
    loop = tqdm(loader)
    model = model.to(device)
    model.train()
    scaler = GradScaler()

    # The following are used to compute the average performance for one epoch, average is computed one single time and not across the batches.
    losses = []
    dice_score_val = []
    iou_score_val = []
    precision_score_val = []
    recall_score_val = []
    specificity_score_val = []
    accuracy_score_val = []
    total_inference_time = 0
    total_frames = 0

    print(f" Current learning rate: {optimizer.param_groups[0]['lr']}")

    # The following loop loads the batches: X will have 16 frames, etc
    for batch_idx, (X, y, info_batch) in enumerate(loop):

        # Save batch metrics
        batch_pred_bin_list = []

        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device) # Ensure proper shape for BCE loss

        # add extra channel to data that only has 2 classes because the model is trained on n_classes=3
        y_n_classes = y.shape[1]

        if y_n_classes < n_classes:
            #  y has shape (B, 2, D, H, W)
            extra_channel = torch.zeros((y.shape[0], 1, y.shape[2], y.shape[3], y.shape[4]), device=y.device)
            y = torch.cat([y, extra_channel], dim=1)  # shape now (B, 3, D, H, W)

        if to_debug:
            print(f"Make sure the shape of the label is proper B,C,D,H,W: {y.shape=}")
            print(f"Make sure the shape of the volume is proper B,C,D,H,W: {X.shape=}")

        # Forward:  measure inference timing
        start_time = time.time()

        with autocast(device_type=device):
            pred = model(X)
            
            if to_debug:
                print(f"Make sure the shape of the prediction is proper B,C,D,H,W: {pred.shape=}")
            loss = loss_fn(pred, y)

        torch.cuda.synchronize()
        end_time = time.time()

        total_inference_time += (end_time-start_time)
        total_frames += X.size(0)

        
        # Loss is computed for the whole batch and a single value is returned. The total of loss values in the list will be the same number as the number of the batches.
        losses.append(loss.item())

        # Mixed precision backward
        scaler.scale(loss).backward()         
        scaler.step(optimizer)                
        scaler.update() 

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

        with torch.no_grad():  # disable gradient tracking for metrics computation

            # Threshold the channel to compare target and prediction. the shape will remain B,C,Z,Y,X 
            pred_bin = (pred > 0.5).long()
            target_bin = (y > 0.5).long()

            if to_debug:
                print(f"""
                    {pred_bin.shape=}
                    {target_bin.shape=}
                    {torch.unique(pred_bin)=}
                    {torch.unique(target_bin)=}

                """)

            # Get batch-wise stats: shapes [batch_size]
            tp, fp, fn, tn = smp_metrics.get_stats(pred_bin, target_bin, num_classes=n_classes, mode='multiclass' )

            # Calculate batch-wise metrics (with reduction)
            iou_metric = smp_metrics.iou_score(tp, fp, fn, tn, reduction='macro')
            dice = smp_metrics.f1_score(tp, fp, fn, tn, reduction='macro')
            accuracy = smp_metrics.accuracy(tp, fp, fn, tn, reduction='macro')
            recall = smp_metrics.recall(tp, fp, fn, tn, reduction='macro')
            specificity = smp_metrics.specificity(tp, fp, fn, tn, reduction='macro')
            precision = smp_metrics.precision(tp, fp, fn, tn, reduction='macro')

            # Store per batch metrics
            dice_score_val.append(dice.item())
            iou_score_val.append(iou_metric.item())
            precision_score_val.append(precision.item())
            recall_score_val.append(recall.item())
            specificity_score_val.append(specificity.item())
            accuracy_score_val.append(accuracy.item())

            if save_preds:
                # Compute per-channel IoU scores (no reduction) for saving
                # the following computes per channel IoU and return [1,3] vals
                frame_iou_vals = smp_metrics.iou_score(tp, fp, fn, tn, reduction='none').detach().cpu().numpy()
                
                batch_pred_bin_list = pred_bin.detach().cpu().float()
                save_predictions(iou_scores=np.array(frame_iou_vals).flatten(), pred=batch_pred_bin_list, volume=X, info_batch=info_batch, y=target_bin, mode='train')

    mean_loss = np.mean(losses)  # Correct loss calculation
    
    # Reported as percentage
    mean_dice = np.mean(dice_score_val) * 100
    mean_iou = np.mean(iou_score_val) * 100
    mean_precision = np.mean(precision_score_val) * 100
    mean_recall = np.mean(recall_score_val) * 100
    mean_specificity = np.mean(specificity_score_val) * 100
    mean_accuracy = np.mean(accuracy_score_val) * 100
    mean_inference_time_per_frame = total_inference_time / total_frames

    return mean_loss, mean_dice, mean_iou, mean_precision, mean_recall, mean_specificity, mean_accuracy, mean_inference_time_per_frame
