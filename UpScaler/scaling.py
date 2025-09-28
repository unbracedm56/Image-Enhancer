import torch
import UpscalerModel
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

def upscale(image, scaler, device=device):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)

    generator = UpscalerModel.Generator(3, 3).to(device)
    generator.load_state_dict(torch.load(f="generator_final_4.pth"))

    torch.cuda.empty_cache()
    generator.eval()
    with torch.inference_mode():
        pred = generator(image.unsqueeze(0).to(device))
    
    output_image = torch.clamp(pred.squeeze().cpu(), 0.0, 1.0) 
    save_image(output_image, "test_output.jpeg")


image = Image.open("low_res_test_image.jpeg")
upscale(image, "2X")