import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class ModelTrainingUtilities:

    @staticmethod # convert to static method if i dont want to instantiate class every time. do not include class self
    def save_predictions_as_imgs(iou_scores, pred, volume, info_batch, y, mode='train', to_debug=False):
        """
        THIS ONLY WORKS FOR BATCH NUM =1 FOR ALL STAGES
        Saves the worst predictions based on IoU scores for 3D volumes (C,Z,Y,X).
        Saves the middle slice along Z for visualization.
        info_batch should contain info for a single volume since batch num=1
        Volume shape is B,C,D,H,W
        Pred shape is B,D,H,W
        y shape is B,D,H,W
        """

        # Ensure results folder exists
        results_folder = "results/"
        os.makedirs(results_folder, exist_ok=True)
        
        case_number = int(info_batch['case_number'])
        dataset_name = str(info_batch['dataset'][0])

        # pick the z slice
        label_vol = y[0,1,:,:,:]  # (D, H, W)

        # random order of z slices
        z_idxs = np.random.permutation(label_vol.shape[0])

        z = None
        for z_idx in z_idxs:
            label_slice = label_vol[z_idx]

            if torch.unique(label_slice).numel() > 1:
                z = z_idx
                break

        # fallback if no valid slice found
        if z is None:
            z = label_vol.shape[0] // 2
            label_slice = label_vol[z]
            
        volume_slice = volume[0, 0, z, :, :]

        # only has class 1
        pred_slice = pred[0, 1, z, :, :]

        if to_debug:
            print(iou_scores, pred.shape, volume.shape, y.shape, info_batch)
            print(f"{volume_slice.shape=} {label_slice.shape=} {pred_slice.shape=}, {np.unique(label_slice.detach().cpu().numpy())=}")

        # Construct filenames
        orig_pathname = f"{mode}_{dataset_name}_{case_number}_slice_{z}_orig.png"
        pred_pathname = f"{mode}_{dataset_name}_{case_number}_slice_{z}_pred.png"
        mask_pathname = f"{mode}_{dataset_name}_{case_number}_slice_{z}_label.png"

        # Save
        torchvision.utils.save_image(volume_slice, os.path.join(results_folder, orig_pathname))
        torchvision.utils.save_image(label_slice.float(), os.path.join(results_folder, mask_pathname))
        torchvision.utils.save_image(pred_slice.float(), os.path.join(results_folder, pred_pathname))

        # print(f"Saved prediction and mask for case {case_number} (IoU: {iou_scores})")


        # Perform overlap analysis
        # Normalize volume for display: only if orig plot is bad
        vol_norm = (volume_slice - volume_slice.min()) / (volume_slice.max() - volume_slice.min() + 1e-8)
        vol = vol_norm.detach().cpu().numpy()
        pred = pred_slice.detach().cpu().numpy()
        label = label_slice.detach().cpu().numpy()

        # Create RGB overlay
        overlay = np.zeros((vol.shape[0], vol.shape[1], 3))

        overlay[..., 0] = pred    # Red channel
        overlay[..., 1] = label   # Green channel
        # Blue stays 0

        # Plot
        fig, ax = plt.subplots()
        ax.imshow(vol, cmap='gray')
        ax.imshow(overlay, alpha=0.5)
        ax.axis('off')
        
        ax.axis('off')

        # Save figure
        overlap_pathname = f"{mode}_{dataset_name}_{case_number}_slice_{z}_overlap.png"
        plt.savefig(os.path.join(results_folder, overlap_pathname), bbox_inches='tight', pad_inches=0)
        plt.close("all")

        # print(f"Saved overlap visualization for case {case_number} (IoU: {iou_scores})")

    @staticmethod
    def save_gradcam_overlay_as_imgs(model, target_layer, pred, volume, info_batch, y, mode='train', results_folder="results/"):
        os.makedirs(results_folder, exist_ok=True)

        case_number = int(info_batch['case_number'])
        dataset_name = str(info_batch['dataset'][0])

        device = volume.device

        # ---------------------------
        # 1. Pick informative slice
        # ---------------------------
        label_vol = y[0]  # (D,H,W)

        z_idxs = np.random.permutation(label_vol.shape[0])
        z = None
        for z_idx in z_idxs:
            if torch.unique(label_vol[z_idx]).numel() > 1:
                z = z_idx
                break

        if z is None:
            z = label_vol.shape[0] // 2

        # ---------------------------
        # 2. Prepare full 3D input
        # ---------------------------
        input_tensor = volume.to(device)  # (1,1,D,H,W)

        # ---------------------------
        # 3. Fix model output (tuple → tensor)
        # ---------------------------
        class GradCAMWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out_main, _ = self.model(x)
                return out_main

        cam_model = GradCAMWrapper(model).to(device)
        cam_model.eval()

        # ---------------------------
        # 4. Proper target mask (3D)
        # ---------------------------
        target_mask = (pred[0] == 1).float().to(device)  # (D,H,W)
        target = [SemanticSegmentationTarget(0, target_mask)]

        # ---------------------------
        # 5. Run Grad-CAM
        # ---------------------------
        with GradCAM(model=cam_model, target_layers=[target_layer]) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=target)

        # ---------------------------
        # 6. Extract slice for visualization
        # ---------------------------
        cam_slice = grayscale_cams[0, z]  # (H,W)

        volume_slice = volume[0, 0, z].detach().cpu().numpy()

        # normalize image to [0,1]
        vol_norm = (volume_slice - volume_slice.min()) / (volume_slice.max() - volume_slice.min() + 1e-8)

        # ---------------------------
        # 7. Overlay
        # ---------------------------
        cam_img = show_cam_on_image(vol_norm, cam_slice, use_rgb=True)

        # ---------------------------
        # 8. Save
        # ---------------------------
        gradcam_pathname = f"{mode}_{dataset_name}_{case_number}_slice_{z}_gradcam.png"
        gradcam_path = os.path.join(results_folder, gradcam_pathname)

        cv2.imwrite(gradcam_path, cam_img)

        print(f"Saved Grad-CAM for case {case_number}, slice {z}")