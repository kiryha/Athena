"""
Image renderer using Stable Diffusion


# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
# Use StableDiffusionXLPipeline for SDXL models (juggernautXL_juggXIByRundiffusion.safetensors)
# Use StableDiffusionPipeline for SD 1.5 models (v1-5-pruned-emaonly-fp16.safetensors)
"""

import time
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler
from PIL import Image

# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors" # SD1.5
MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors" # SDXL

# CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/control_v11p_sd15_canny_fp16.safetensors" # SD1.
CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/diffusion_pytorch_model_canny.fp16.safetensors" # SDXL

width=832   
height=1216


# Load MAIN pipeline
pipeline = StableDiffusionXLPipeline.from_single_file(MODEL_PATH, torch_dtype=torch.float16)
pipeline.to("cuda")

# Load ControlNet pipeline
controlnet = ControlNetModel.from_single_file(CONTROLNET_PATH, torch_dtype=torch.float16)
controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_single_file(MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16)
# Load scheduler
controlnet_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    controlnet_pipeline.scheduler.config,
    algorithm_type="sde-dpmsolver++",  # DPM++ SDE
    use_karras_sigmas=True              # Karras
)
controlnet_pipeline.to("cuda")


def render_image(prompt: str, steps: int, seed: int, cfg: float, controlnet_strength: float,
                 output_path: str, control_image_path: str = None):
    
    # Mesure render time
    start_time = time.perf_counter()

    # Define seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Render image
    if control_image_path:
        control_image = Image.open(control_image_path).convert("RGB")
        image = controlnet_pipeline(prompt, image=control_image, num_inference_steps=steps,
                     generator=generator, controlnet_conditioning_scale=controlnet_strength, 
                     guidance_scale=cfg,
                     width=width, height=height).images[0]
    else:
        image = pipeline(prompt, num_inference_steps=steps, generator=generator, 
                        guidance_scale=cfg,
                        width=width, height=height).images[0]
    # Save image
    image.save(output_path)
    
    # Mesure render time and format as MM:SS
    render_time = time.perf_counter() - start_time
    minutes = int(render_time // 60)
    seconds = int(render_time % 60)
    return f"{minutes:02d}:{seconds:02d}"