import torch
from tqdm import tqdm
import numpy as np
from config import *
import segmentation_models_pytorch.metrics as smp_metrics 
import time
from torch.amp import autocast, GradScaler
from model_training_utilities import ModelTrainingUtilities
from monai.inferers import SlidingWindowInferer

save_predictions = ModelTrainingUtilities.save_predictions_as_imgs

def val_fn(loss_fn, loader, model, device='cuda', save_preds=True):
    
    # set the sliding window to obtian 96^3 patches of validation volumes, compute predictions, stitch back predictions, and return the stiched prediction
    inferer = SlidingWindowInferer(
        roi_size=(chunk_z, chunk_y, chunk_x),
        sw_batch_size=2,
        overlap=0.75,  # corresponds to stride of 24
        mode="gaussian"  # smooth blending across overlapping patches
        )

    loop = tqdm(loader)
    model = model.to(device)
    model.eval() # in evaluation mode

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

    # The following loop loads the batches: X will have 16 frames, etc
    for batch_idx, (X, y, info_batch) in enumerate(loop):

        # wrap in torch no grad:
        with torch.no_grad():
            # Save batch metrics
            batch_pred_bin_list = []

            X = X.to(device) # the batch size for validaiton is 1 and the X shape should be 1,1, 160,160,160

            print(f"Make sure the X shape is 1,C,Z,Y,X: {X.shape=}")

            y = y.to(device) # Ensure proper shape for BCE loss
            print(f"Make sure the shape of the label is proper B,C,D,H,W: {y.shape=}")

            # Forward:  measure inference timing
            start_time = time.time()

            preds = inferer(inputs=X, network=model)
            loss = loss_fn(preds, y)

            torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time-start_time)
            total_frames += X.size(0)

            
            # Loss is computed for the whole batch and a single value is returned. The total of loss values in the list will be the same number as the number of the batches.
            losses.append(loss.item())


            # Update tqdm loop
            loop.set_postfix(loss=loss.item())
            # Threshold the channel to compare target and prediction. the shape will change B,C,Z,Y,X -> B,Z,Y,X and loses the one-hot encoding
            pred_bin = torch.argmax(preds, dim=1)
            target_bin = torch.argmax(y, dim=1)

            # Get batch-wise stats: shapes [batch_size]
            tp, fp, fn, tn = smp_metrics.get_stats(pred_bin, target_bin, mode='multiclass')

            # Calculate batch-wise metrics (with reduction)
            iou_metric = smp_metrics.iou_score(tp, fp, fn, tn, reduction='micro')
            dice = smp_metrics.f1_score(tp, fp, fn, tn, reduction='micro')
            accuracy = smp_metrics.accuracy(tp, fp, fn, tn, reduction='micro')
            recall = smp_metrics.recall(tp, fp, fn, tn, reduction='micro')
            specificity = smp_metrics.specificity(tp, fp, fn, tn, reduction='micro')
            precision = smp_metrics.precision(tp, fp, fn, tn, reduction='micro')

            # Store per batch metrics
            dice_score_val.append(dice.item())
            iou_score_val.append(iou_metric.item())
            precision_score_val.append(precision.item())
            recall_score_val.append(recall.item())
            specificity_score_val.append(specificity.item())
            accuracy_score_val.append(accuracy.item())

            if save_preds:
                # Compute per-frame IoU scores (no reduction) for saving
                frame_iou_vals = smp_metrics.iou_score(tp, fp, fn, tn, reduction='none').detach().cpu().numpy()
                
                batch_pred_bin_list = pred_bin.detach().cpu().float()
                save_predictions(np.array(frame_iou_vals).flatten(), batch_pred_bin_list, X, info_batch, y, mode='val')

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
