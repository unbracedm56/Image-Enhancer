import streamlit as st
from PIL import Image
# NEW DEPENDENCY: You must install this: pip install streamlit-cropper
from streamlit_cropper import st_cropper 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from huggingface_hub import hf_hub_download
import tensorflow as tf
import io
import math
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
import cv2

# --- Streamlit Configuration ---
st.set_page_config(layout="wide", page_title="Image Model Showcase")
# WARNING FIX: Removed obsolete st.set_option('deprecation.showPyplotGlobalUse', False)
# To suppress persistent TensorFlow/Keras warnings, run your app with: export TF_CPP_MIN_LOG_LEVEL=2

# --- Model 1: Image Colorization ---

# Define the U-Net Generator model architecture for colorization
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

@st.cache_resource
def get_colorization_model():
    """Downloads and caches the colorization model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Downloading colorization model... This may take a moment."):
        repo_id = "kp7575/grayscale_to_color"
        filename = "generator_epoch_17.pth"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def process_colorization(model, device, input_image):
    """Converts a grayscale PIL image to a colorized PIL image."""
    # The model expects a 512x512 input, which is handled via st_cropper before calling this function
    img_resized = transforms.Resize((512, 512), Image.BICUBIC)(input_image.convert("RGB"))
    img_lab = rgb2lab(np.array(img_resized)).astype("float32")
    lab_tensor = torch.from_numpy(img_lab).permute(2, 0, 1)
    L = lab_tensor[[0], ...] / 50.0 - 1.0
    L = L.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_ab = model(L).cpu()
    L_denormalized = (L.cpu() + 1.0) * 50.0
    predicted_ab_denormalized = predicted_ab * 128.0
    colorized_lab_tensor = torch.cat([L_denormalized, predicted_ab_denormalized], dim=1).squeeze(0)
    colorized_lab_numpy = colorized_lab_tensor.permute(1, 2, 0).numpy()
    colorized_rgb = lab2rgb(colorized_lab_numpy)
    final_image = (colorized_rgb * 255).astype(np.uint8)
    return Image.fromarray(final_image)

# --- Model 2: Image Brightening ---

# Define the custom Instance Normalization layer for the brightening model
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],), initializer="ones", trainable=True)
        self.beta  = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=True)
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1,2], keepdims=True)
        return self.gamma * (x - mean)/tf.sqrt(var + self.epsilon) + self.beta

@st.cache_resource
def get_brightening_model():
    """Downloads and caches the image brightening model."""
    with st.spinner("Downloading brightening model... This may take a moment."):
        HF_MODEL_ID = "Coder-M/Bright"
        MODEL_FILE = hf_hub_download(repo_id=HF_MODEL_ID, filename="bright_gan_generator_epoch45.h5")
    # Must use custom_objects for custom layers like InstanceNormalization
    generator = tf.keras.models.load_model(MODEL_FILE, compile=False,
                                           custom_objects={'InstanceNormalization': InstanceNormalization})
    return generator

def process_brightening(generator, input_image):
    """Brightens a dark PIL image. The model expects a 256x256 input."""
    img = input_image.convert("RGB")
    img_array = np.array(img)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # The image is expected to be 256x256, but the resize is done here if needed
    img_tensor = tf.image.resize(img_tensor, [256, 256]) 
    img_tensor = (img_tensor / 127.5) - 1.0
    img_input = tf.expand_dims(img_tensor, 0)
    pred = generator(img_input, training=False)
    pred_img = (pred[0].numpy() + 1) / 2.0
    pred_img = (pred_img * 255).astype(np.uint8)
    return Image.fromarray(pred_img)

# --- Model 3: Neural Style Transfer ---

# Define the VGG model architecture for style transfer
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

@st.cache_resource
def get_style_transfer_model():
    """Loads and caches the VGG model for style transfer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG().to(device).eval()
    return model, device

def process_style_transfer(vgg_model, device, content_image, style_image, total_steps, learning_rate, alpha, beta):
    """Performs neural style transfer. The style image is resized to match the content image's dimensions."""
    
    # 1. Define a simple loader just for ToTensor
    loader = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 2. Load content image. This defines the output size.
    content_tensor = loader(content_image.convert("RGB")).unsqueeze(0).to(device)
    
    # 3. Load style image (unresized)
    style_tensor_unresized = loader(style_image.convert("RGB")).unsqueeze(0).to(device)
    
    # 4. Resize style_tensor to match content_tensor shape
    #    This is the new logic you requested.
    style_tensor = transforms.functional.resize(
        style_tensor_unresized.squeeze(0), 
        content_tensor.shape[2:], # Use (H, W) from content tensor
        interpolation=transforms.InterpolationMode.BICUBIC # Added for quality
    ).unsqueeze(0).to(device)

    # 5. Clone content_tensor to create the generated image
    generated = content_tensor.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=learning_rate)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    image_placeholder = st.empty()

    for step in range(total_steps):
        generated_features = vgg_model(generated)
        original_img_features = vgg_model(content_tensor)
        style_features = vgg_model(style_tensor)

        style_loss = original_loss = 0
        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            _, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)
            G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
            A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            progress = (step + 1) / total_steps
            progress_bar.progress(progress)
            status_text.text(f"Step [{step+1}/{total_steps}] | Loss: {total_loss.item():.4f}")
            
            img_to_show = generated.clone().squeeze(0).cpu().detach()
            img_to_show = transforms.ToPILImage()(img_to_show)
            image_placeholder.image(img_to_show, caption=f"Generated Image (Step {step+1})")

    progress_bar.empty()
    status_text.empty()
    final_img_tensor = generated.squeeze(0).cpu().detach()
    final_img = transforms.ToPILImage()(final_img_tensor)
    return final_img

# --- Model 4: SwinIR Image Denoising ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                # Assuming 'checkpoint' is a module from a standard library (e.g., torch.utils.checkpoint)
                # Since it's not defined, I will comment out the checkpoint usage
                # x = checkpoint.checkpoint(blk, x, x_size)
                x = blk(x, x_size) # Fallback to regular forward
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x is B C H W
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x # B N C


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        # x is B N C
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x # B C H W


class SwinIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = x + self.conv_last(res)
        return x[:, :, :H*self.upscale, :W*self.upscale]


@st.cache_resource
def get_denoising_model():
    """Downloads and caches the SwinIR denoising model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with st.spinner("Downloading denoising model... This may take a moment."):
        model_path = hf_hub_download(
            repo_id="Aion365/denoising_images",
            filename="005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
        )
    # SwinIR model for Denoising (upscale=1)
    model = SwinIR(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'
    pretrained_model = torch.load(model_path, map_location=device)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)
    return model, device

def process_denoising(model, device, input_image):
    """Denoises a noisy PIL image using the SwinIR model."""
    window_size = 8
    scale = 1 # Denoising is 1x scale
    
    img_lq = np.array(input_image).astype(np.float32) / 255.
    
    if img_lq.ndim == 3:
        img_lq = np.transpose(img_lq, (2, 0, 1))
    
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()
        # The model requires image dimensions to be multiples of window_size (8)
        h_pad = (window_size - h_old % window_size) % window_size
        w_pad = (window_size - w_old % window_size) % window_size
        img_lq = F.pad(img_lq, (0, w_pad, 0, h_pad), 'reflect')

        output = model(img_lq)
        output = output[..., :h_old * scale, :w_old * scale]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output, (1, 2, 0)) # CHW-RGB to HWC-RGB
    output = (output * 255.0).round().astype(np.uint8)
    
    return Image.fromarray(output)

# --- Model 5: Image Upscaling ---

class DenseBlock(nn.Module):
  def __init__(self, in_channels, out_channels = 32):
    super().__init__()
    self.layers = nn.ModuleList()
    self.in_channels = in_channels
    self.out_channels = out_channels

    for i in range(5):
      self.layers.append(
          nn.Sequential(
              nn.Conv2d(self.in_channels + i * self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
              nn.LeakyReLU(0.2, inplace=True)
          )
      )

    self.conv1 = nn.Conv2d(self.in_channels + 5 * self.out_channels, self.in_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    features = [x]
    for layer in self.layers:
      out = layer(torch.cat(features, dim=1))
      features.append(out)
    out = self.conv1(torch.cat(features, dim=1))
    return out


class RRDB(nn.Module):  #Residual-in-Residual Dense Block
  def __init__(self, in_channels, out_channels = 32, beta=0.2):
    super().__init__()
    self.beta = beta
    self.block1 = DenseBlock(64, 32)
    self.block2 = DenseBlock(64, 32)
    self.block3 = DenseBlock(64, 32)

  def forward(self, x):
    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    return out * self.beta + x

class UpscalerGenerator(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv_initial = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
    self.RRDB_sequence = nn.ModuleList()

    for i in range(8):
      self.RRDB_sequence.append(
          RRDB(64, 32, 0.2)
      )

    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    self.Upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    self.image_reconstruction = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, x):
    x_initial = self.conv_initial(x)
    out = x_initial
    for layer in self.RRDB_sequence:
      out = layer(out)
    out = self.conv2(out)
    out = x_initial + out
    out = self.Upsampling(out)
    out = self.image_reconstruction(out)
    return out

@st.cache_resource
def get_upscaler_model():
    """Downloads and caches the upscaling model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Downloading upscaler model... This may take a moment."):
        generator_path = hf_hub_download(repo_id="unbracedm56/Upscaler2x", filename="generator_final_4.pth")
    model = UpscalerGenerator(3, 3).to(device)
    model.load_state_dict(torch.load(f=generator_path, map_location=device))
    model.eval()
    return model, device

def process_upscaling(model, device, input_image):
    """Upscales a low-resolution PIL image."""
    # The upscaling model can handle various input sizes, but processing a cropped
    # image will still be faster and apply the upscale to that region.
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
    
    output_tensor = torch.clamp(pred.squeeze().cpu(), 0.0, 1.0) 
    output_image = transforms.ToPILImage()(output_tensor)
    return output_image


# --- Streamlit UI ---

st.title("🎨 Image Model Showcase")
st.sidebar.title("Controls")
model_choice = st.sidebar.selectbox("Choose a model:", (
    "Image Colorization", 
    "Image Brightening", 
    "Neural Style Transfer", 
    "Image Denoising",
    "Image Upscaling"
    ))
st.sidebar.markdown("---")

# --- UI for Image Colorization ---
if model_choice == "Image Colorization":
    st.header("Image Colorization")
    st.info("Upload a black and white image. The model requires a **512x512** input. Use the cropper to select the area you want to colorize.")
    uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.subheader("🎨 Interactive Crop (1:1 Aspect Ratio)")
        st.warning("Please drag the box to select the area for colorization. The cropped image will be displayed on the right.")
        
        # Interactive Cropper for 1:1 aspect ratio (which is 512x512)
        cropped_image = st_cropper(
            original_image, 
            aspect_ratio=(1, 1), 
            box_color='yellow', 
            realtime_update=True, 
            key="color_cropper"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption=f"Original Image ({original_image.width}x{original_image.height})", use_container_width=True) # WARNING FIX
            if st.button("🎨 Colorize Selected Crop"):
                model, device = get_colorization_model()
                with st.spinner("Applying color..."):
                    # The image output from st_cropper is already cropped (cropped_image)
                    colorized_image = process_colorization(model, device, cropped_image)
                with col2:
                    st.image(colorized_image, caption="Colorized Image (512x512 Result)", use_container_width=True) # WARNING FIX
        with col2:
             st.image(cropped_image, caption=f"Selected Crop Preview (Will be resized to 512x512)", use_container_width=True) # WARNING FIX

# --- UI for Image Brightening ---
elif model_choice == "Image Brightening":
    st.header("Image Brightening")
    st.info("Upload a dark image. The model is trained for **256x256** input. Use the cropper to select the area you want to brighten.")
    uploaded_file = st.file_uploader("Choose a dark image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("💡 Interactive Crop (1:1 Aspect Ratio)")
        st.warning("Please drag the box to select the area for brightening. The cropped image will be displayed on the right.")

        # Interactive Cropper for 1:1 aspect ratio (which is 256x256)
        cropped_image = st_cropper(
            original_image, 
            aspect_ratio=(1, 1), 
            box_color='blue', 
            realtime_update=True, 
            key="bright_cropper"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption=f"Original Image ({original_image.width}x{original_image.height})", use_container_width=True) # WARNING FIX
            if st.button("💡 Brighten Selected Crop"):
                model = get_brightening_model()
                with st.spinner("Illuminating image..."):
                    brightened_image = process_brightening(model, cropped_image)
                with col2:
                    st.image(brightened_image, caption="Brightened Image (256x256 Result)", use_container_width=True) # WARNING FIX
        with col2:
            st.image(cropped_image, caption=f"Selected Crop Preview (Will be resized to 256x256)", use_container_width=True) # WARNING FIX


# --- UI for Neural Style Transfer ---
elif model_choice == "Neural Style Transfer":
    st.header("Neural Style Transfer")
    # --- CHANGED ---
    st.info("Transfer the style of one image onto the content of another. The style image will be resized to match the content image's dimensions.")
    
    col1, col2 = st.columns(2)
    with col1:
        content_file = st.file_uploader("Upload Content Image...", type=["jpg", "jpeg", "png"])
    with col2:
        style_file = st.file_uploader("Upload Style Image...", type=["jpg", "jpeg", "png"])

    if content_file is not None and style_file is not None:
        content_image = Image.open(content_file)
        style_image = Image.open(style_file)
        
        # --- CHANGED ---
        st.subheader("Your Images")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            # --- CHANGED ---
            st.image(content_image, caption=f"Content Image ({content_image.width}x{content_image.height})", use_container_width=True)
        with img_col2:
            # --- CHANGED ---
            st.image(style_image, caption=f"Style Image ({style_image.width}x{style_image.height})", use_container_width=True)

        st.sidebar.markdown("### Style Transfer Settings")
        total_steps = st.sidebar.slider("Training Steps", 500, 5000, 1000, 100)
        learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.002, format="%.4f")
        alpha = st.sidebar.slider("Content Weight (α)", 0.1, 10.0, 1.0, 0.1)
        beta = st.sidebar.slider("Style Weight (β)", 0.01, 2.0, 0.05, 0.01)

        if st.button("🚀 Start Style Transfer"):
            st.subheader("Generated Image")
            model, device = get_style_transfer_model()
            generated_image = process_style_transfer(model, device, content_image, style_image, total_steps, learning_rate, alpha, beta)
            st.success("Style transfer complete!")
            st.image(generated_image, caption="Final Generated Image", use_container_width=True) # WARNING FIX

# --- UI for Image Denoising ---
elif model_choice == "Image Denoising":
    st.header("Image Denoising")
    st.info("Upload a noisy image. Large images are slow, but you can use the cropper to select a portion for faster processing.")
    
    st.sidebar.markdown("### Denoising Processing Options")
    crop_denoise_option = st.sidebar.radio("Processing Mode:", ("Process Full Image", "Select Crop to Process"))

    uploaded_file = st.file_uploader("Choose a noisy image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        image_to_process = original_image.copy()

        if crop_denoise_option == "Select Crop to Process":
            st.subheader("🧹 Interactive Crop for Denoising")
            st.warning("Please drag the box to select the area for denoising.")
            
            cropped_image = st_cropper(
                original_image, 
                aspect_ratio=None, 
                box_color='yellow', 
                realtime_update=True, 
                key="denoise_cropper"
            )
            image_to_process = cropped_image
            st.info(f"Processing a cropped image of size {image_to_process.width}x{image_to_process.height}.")
            
        else:
            st.subheader("Original Image")
            st.info(f"Processing the full image of size {original_image.width}x{original_image.height}. This may be slow.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_to_process, caption="Image to Denoise", use_container_width=True) # WARNING FIX

        if st.button("🧹 Denoise Image"):
            model, device = get_denoising_model()
            with st.spinner("Removing noise... This may take some time."):
                denoised_image = process_denoising(model, device, image_to_process)
            
            with col2:
                st.image(denoised_image, caption="Denoised Image", use_container_width=True) # WARNING FIX

# --- UI for Image Upscaling (MODIFIED) ---
elif model_choice == "Image Upscaling":
    st.header("Image Upscaling")
    st.info("Upload a low-resolution image to increase its quality and size (2x). Use the cropper to select a portion to speed up the process.")
    st.warning("Upscaling is computationally heavy. Processing the full image can be very slow.")
    
    st.sidebar.markdown("### Upscaling Processing Options")
    crop_upscale_option = st.sidebar.radio("Processing Mode:", ("Process Full Image", "Select Crop to Process"))
    
    uploaded_file = st.file_uploader("Choose a low-resolution image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        image_to_process = original_image.copy()
        
        if crop_upscale_option == "Select Crop to Process":
            st.subheader("🚀 Interactive Crop for Upscaling")
            st.warning("Please drag the box to select the area for upscaling.")
            
            cropped_image = st_cropper(
                original_image, 
                aspect_ratio=None, 
                box_color='blue', 
                realtime_update=True, 
                key="upscale_cropper"
            )
            image_to_process = cropped_image
            st.info(f"Processing a cropped image of size {image_to_process.width}x{image_to_process.height}.")
        
        else:
            st.subheader("Original Image")
            st.info(f"Processing the full image of size {original_image.width}x{original_image.height}. This may be very slow.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_to_process, caption=f"Image to Upscale ({image_to_process.width}x{image_to_process.height})", use_container_width=True) # WARNING FIX

        if st.button("🚀 Upscale Image"):
            model, device = get_upscaler_model()
            with st.spinner("Upscaling image... Please be patient, this will take a while."):
                upscaled_image = process_upscaling(model, device, image_to_process)

            with col2:
                st.image(upscaled_image, caption=f"Upscaled Image (Result is 2x the input size: {upscaled_image.width}x{upscaled_image.height})", use_container_width=True) # WARNING FIX
