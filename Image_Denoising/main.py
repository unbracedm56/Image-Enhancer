import numpy as np
import os
import torch
import torch.nn.functional as F
import cv2  # <-- Added missing import
from huggingface_hub import hf_hub_download
from collections import OrderedDict

# Assuming network_swinir.py and util_calculate_psnr_ssim.py are in the same directory
from network_swinir import SwinIR as net
import util_calculate_psnr_ssim as util


def main():
    # --- 1. SETTINGS AND MODEL DOWNLOAD ---
    # These settings are specific to the model '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth'
    task = 'color_dn'
    noise = 50
    training_patch_size = 128
    window_size = 8
    scale = 1
    
    # Download the pre-trained model weights
    model_path = hf_hub_download(
        repo_id="Aion365/denoising_images",
        filename="005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
    )

    # Set up the device (use CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2. DEFINE AND LOAD THE MODEL ---
    # Define the model architecture
    model = net(upscale=1, in_chans=3, img_size=training_patch_size, window_size=window_size,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    
    # Load the downloaded weights
    param_key_g = 'params'
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
    # Prepare the model for inference
    model.eval()
    model = model.to(device)

    # --- 3. IMAGE INPUT AND OUTPUT ---
    img_path = input("Enter the path to your noisy image: ")
    if not os.path.exists(img_path):
        print("Error: Image path does not exist.")
        return
        
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)

    # --- 4. READ AND PRE-PROCESS IMAGE ---
    # Read the low-quality (noisy) image
    imgname, _ = os.path.splitext(os.path.basename(img_path))
    img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    
    # Convert from HWC-BGR to NCHW-RGB tensor
    img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # --- 5. INFERENCE AND PADDING ---
    with torch.no_grad():
        # Get original image size
        _, _, h_old, w_old = img_lq.size()

        # Pad the image to be divisible by the window size
        h_pad = (window_size - h_old % window_size) % window_size
        w_pad = (window_size - w_old % window_size) % window_size
        img_lq = F.pad(img_lq, (0, w_pad, 0, h_pad), 'reflect')

        # Run the model
        output = model(img_lq)

        # Crop the output back to the original size
        output = output[..., :h_old * scale, :w_old * scale]

    # --- 6. SAVE THE RESULT ---
    # Convert tensor back to a savable image format
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    # Save the denoised image
    save_path = f'{save_dir}/{imgname}_SwinIR_denoised.png'
    cv2.imwrite(save_path, output)

    print(f"Done! Denoised image saved to: {save_path}")


if __name__ == '__main__':
    main()