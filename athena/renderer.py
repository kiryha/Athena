"""
Image renderer using Stable Diffusion


# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
# Use StableDiffusionXLPipeline for SDXL models (juggernautXL_juggXIByRundiffusion.safetensors)
# Use StableDiffusionPipeline for SD 1.5 models (v1-5-pruned-emaonly-fp16.safetensors)
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"


pipeline = StableDiffusionPipeline.from_single_file(MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
pipeline.to("cuda")


def render_image(prompt: str, steps: int, seed: int, output_path: str):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipeline(prompt, num_inference_steps=steps, generator=generator).images[0]
    image.save(output_path)
