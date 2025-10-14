import torch
import UpscalerModel
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def upscale(image, scaler, device=device):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)

    generator_path = hf_hub_download(repo_id="unbracedm56/Upscaler2x", filename="generator_final_4.pth", token=hf_token)
    generator = UpscalerModel.Generator(3, 3).to(device)
    generator.load_state_dict(torch.load(f=generator_path))

    torch.cuda.empty_cache()
    generator.eval()
    with torch.inference_mode():
        pred = generator(image.unsqueeze(0).to(device))
    
    output_image = torch.clamp(pred.squeeze().cpu(), 0.0, 1.0) 
    save_image(output_image, "test_output.jpeg")

