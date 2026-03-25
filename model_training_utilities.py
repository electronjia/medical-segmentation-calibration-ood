import os
import numpy as np
import torchvision
import torch
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class ModelTrainingUtilities:
    def __init__(self):
        pass


    def save_predictions_as_imgs(self, iou_scores, preds, frames, info_batch, y, mode='train'):
        """
        Saves the worst predictions based on IoU scores for 3D volumes (C,Z,Y,X).
        Saves the middle slice along Z for visualization.
        """

        # Ensure results folder exists
        results_folder = "results/"
        os.makedirs(results_folder, exist_ok=True)

        # Get indices of worst predictions
        worst_indices = np.argsort(iou_scores)

        for i in worst_indices:
            info_indiv = info_batch[i]
            case_number = info_indiv['case_number']
            dataset_name = info_indiv['dataset']

            # Pick middle slice along Z
            z = frames[i].shape[1] // 2
            frame_slice = frames[i][:, z, :, :]
            mask_slice = y[i][:, z, :, :]
            pred_slice = preds[i][:, z, :, :]

            # Construct filenames
            orig_pathname = f"{mode}_{dataset_name}_{case_number}_orig.png"
            pred_pathname = f"{mode}_{dataset_name}_{case_number}_pred.png"
            mask_pathname = f"{mode}_{dataset_name}_{case_number}_mask.png"

            # Save
            torchvision.utils.save_image(frame_slice, os.path.join(results_folder, orig_pathname))
            torchvision.utils.save_image(mask_slice, os.path.join(results_folder, mask_pathname))
            torchvision.utils.save_image(pred_slice, os.path.join(results_folder, pred_pathname))

            print(f"Saved prediction and mask for case {case_number} (IoU: {iou_scores[i]:.2f})")

    def save_gradcam_overlay_as_imgs(model, target_layer, masks, frames, info_batch, mode='train', results_folder="results/"):
        """
        Generates and saves GradCAM images for 3D medical volumes by applying slice-wise Grad-CAM.

        Args:
            model: nn.Module
            target_layer: layer to compute Grad-CAM
            masks: Tensor, shape (N, C, Z, Y, X)
            frames: Tensor, shape (N, C, Z, Y, X)
            info_batch: list of dicts containing metadata (case_number, dataset)
            mode: 'train' or 'val' for naming
            results_folder: folder to save Grad-CAM images
        """

        os.makedirs(results_folder, exist_ok=True)

        for frame, mask, info_indiv in zip(frames, masks, info_batch):
            case_number = info_indiv['case_number']
            dataset_name = info_indiv['dataset']

            # Pick the middle slice along Z axis
            z = frame.shape[2] // 2  # frame shape: (C, Z, Y, X)
            img_2d = frame[:, z, :, :]       # (C, Y, X)
            mask_2d = mask[:, z, :, :]       # (C, Y, X)

            # Convert to H, W, C for visualization
            rgb_img = img_2d.permute(1, 2, 0).cpu().numpy()  # (Y, X, C)
            mask_np = mask_2d.squeeze(0).cpu().numpy()       # (Y, X), assume channel=1 for target

            # Prepare input tensor (N, C, H, W)
            input_tensor = rgb_img.transpose(2, 0, 1)[None, :, :, :]  # (1, C, H, W)
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(frame.device)

            # Set target for segmentation
            target = [SemanticSegmentationTarget(0, mask_np)]

            # Run Grad-CAM
            with GradCAM(model=model, target_layers=[target_layer], use_cuda=frame.is_cuda) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=target)

            # Overlay CAM on image
            cam_img = show_cam_on_image(rgb_img, grayscale_cams[0, :], use_rgb=True)

            # Construct save path
            gradcam_pathname = f"{mode}_{dataset_name}_{case_number}_gradcam.png"
            gradcam_path = os.path.join(results_folder, gradcam_pathname)

            # Save
            cv2.imwrite(gradcam_path, cam_img)