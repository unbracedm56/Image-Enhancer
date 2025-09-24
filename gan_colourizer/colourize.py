import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import argparse

from huggingface_hub import hf_hub_download


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.down1 = self.contract_block(in_channels, 64, normalize=False)
        self.down2 = self.contract_block(64, 128)
        self.down3 = self.contract_block(128, 256)
        self.down4 = self.contract_block(256, 512)
        self.down5 = self.contract_block(512, 512)
        self.up1 = self.expand_block(512, 512)
        self.up2 = self.expand_block(1024, 256)
        self.up3 = self.expand_block(512, 128)
        self.up4 = self.expand_block(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def contract_block(self, in_c, out_c, k=4, s=2, p=1, normalize=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def expand_block(self, in_c, out_c, k=4, s=2, p=1):
        layers = [nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
                  nn.BatchNorm2d(out_c), nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))
        return self.final(torch.cat([u4, d1], 1))


def colorize_image(input_path, output_path):
    """Loads a trained generator from Hugging Face, colorizes an image, and saves the result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    repo_id = "kp7575/grayscale_to_color" 
    filename = "generator_epoch_17.pth"   

    print(f"Downloading model from Hugging Face Hub: {repo_id}/{filename}")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Model successfully downloaded to: {model_path}")
   

    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    img = Image.open(input_path).convert("RGB")
    img_resized = transforms.Resize((512, 512), Image.BICUBIC)(img)
    img_lab = rgb2lab(np.array(img_resized)).astype("float32")
    lab_tensor = torch.from_numpy(img_lab).permute(2, 0, 1)

    L = lab_tensor[[0], ...] / 50.0 - 1.0
    L = L.unsqueeze(0).to(device)

    print("Colorizing image...")
    with torch.no_grad():
        predicted_ab = model(L).cpu()

    L_denormalized = (L.cpu() + 1.0) * 50.0
    predicted_ab_denormalized = predicted_ab * 128.0

    colorized_lab_tensor = torch.cat([L_denormalized, predicted_ab_denormalized], dim=1).squeeze(0)
    colorized_lab_numpy = colorized_lab_tensor.permute(1, 2, 0).numpy()
    colorized_rgb = lab2rgb(colorized_lab_numpy)

    final_image = (colorized_rgb * 255).astype(np.uint8)
    Image.fromarray(final_image).save(output_path)
    print(f"Successfully saved colorized image to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Colorize a black and white image using a trained U-Net model downloaded from Hugging Face Hub.")
   
    parser.add_argument('--input', type=str, required=True, help="Path to the input black and white image.")
    parser.add_argument('--output', type=str, default="colorized_output.jpg", help="Path to save the colorized output image.")
    args = parser.parse_args()

    
    colorize_image(input_path=args.input, output_path=args.output)

