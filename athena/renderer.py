"""
Image renderer using Stable Diffusion


# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
# Use StableDiffusionXLPipeline for SDXL models (juggernautXL_juggXIByRundiffusion.safetensors)
# Use StableDiffusionPipeline for SD 1.5 models (v1-5-pruned-emaonly-fp16.safetensors)
"""

import time
import torch
from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline,
    ControlNetModel, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler)
from PIL import Image

MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors" # SD1.5
# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors" # SDXL

CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/control_v11p_sd15_scribble_fp16.safetensors" # SD1.5
# CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/diffusion_pytorch_model_canny.fp16.safetensors" # SDXL

width=512
height=512


# # Load MAIN pipeline
# pipeline = StableDiffusionPipeline.from_single_file(MODEL_PATH, torch_dtype=torch.float16)
# pipeline.to("cuda")

# Load ControlNet pipeline
controlnet = ControlNetModel.from_single_file(CONTROLNET_PATH, torch_dtype=torch.float16)
controlnet_pipeline = StableDiffusionControlNetPipeline.from_single_file(
    MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16)
controlnet_pipeline.to("cuda")

controlnet_pipeline.enable_attention_slicing()
controlnet_pipeline.enable_vae_slicing()
# controlnet_pipeline.enable_xformers_memory_efficient_attention()

print("---------------------------------")
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(next(controlnet_pipeline.unet.parameters()).device)
print("---------------------------------")

def get_scheduler(pipe, sampler_name: str):
    """Configure scheduler based on sampler name."""
    config = pipe.scheduler.config
    
    if sampler_name == "DPM++ 2M":
        return DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
            
    elif sampler_name == "DPM++ 2M SDE":
        return DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)

    elif sampler_name == "DPM++ 2S a":
        return DPMSolverSinglestepScheduler.from_config(
            config, use_karras_sigmas=True)

    elif sampler_name == "Euler":
        return EulerDiscreteScheduler.from_config(config)
    
    elif sampler_name == "Euler A":
        return EulerAncestralDiscreteScheduler.from_config(config)
   
    else:
        # Default to DPM++ 2M
        return DPMSolverMultistepScheduler.from_config(
            config, algorithm_type="dpmsolver++", use_karras_sigmas=True)


def render_image(prompt: str, negative_prompt: str, steps: int, seed: int, cfg: float,
                 sampler: str, controlnet_strength: float, output_path: str,
                 control_image_path: str = None):
    
    # Measure render time
    start_time = time.perf_counter()

    # Define seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Render image
    if control_image_path:
        control_image = Image.open(control_image_path).convert("RGB")
        # Set scheduler
        controlnet_pipeline.scheduler = get_scheduler(controlnet_pipeline, sampler)
        image = controlnet_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=control_image,
            num_inference_steps=steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_strength,
            guidance_scale=cfg,
            width=width,
            height=height).images[0]
    else:
        # Set scheduler
        pipeline.scheduler = get_scheduler(pipeline, sampler)
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=cfg,
            width=width,
            height=height).images[0]
    
    # Save image
    image.save(output_path)
    
    # Measure render time and format as MM:SS
    render_time = time.perf_counter() - start_time
    minutes = int(render_time // 60)
    seconds = int(render_time % 60)
    return f"{minutes:02d}:{seconds:02d}"