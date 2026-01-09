"""
Image renderer using Stable Diffusion


# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
# Use StableDiffusionXLPipeline for SDXL models (juggernautXL_juggXIByRundiffusion.safetensors)
# Use StableDiffusionPipeline for SD 1.5 models (v1-5-pruned-emaonly-fp16.safetensors)
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/control_v11p_sd15_canny_fp16.safetensors"

width=512
height=512


# Load MAIN pipeline
pipeline = StableDiffusionPipeline.from_single_file(MODEL_PATH, torch_dtype=torch.float16)
pipeline.to("cuda")

# Load ControlNet pipeline
controlnet = ControlNetModel.from_single_file(CONTROLNET_PATH, torch_dtype=torch.float16)
controlnet_pipeline = StableDiffusionControlNetPipeline.from_single_file(MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16)
controlnet_pipeline.to("cuda")


def render_image(prompt: str, steps: int, seed: int, output_path: str,
                 control_image_path: str = None, controlnet_weight: float = 1.0):
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    if control_image_path:
        control_image = Image.open(control_image_path).convert("RGB")
        image = controlnet_pipeline(prompt, image=control_image, num_inference_steps=steps,
                     generator=generator, controlnet_conditioning_scale=controlnet_weight, 
                     width=width, height=height).images[0]
    else:
        image = pipeline(prompt, num_inference_steps=steps, generator=generator, 
                        width=width, height=height).images[0]
    
    image.save(output_path)
