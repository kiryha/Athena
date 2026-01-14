"""
Image renderer using Stable Diffusion

pip install diffusers transformers accelerate torchao sentencepiece imageio-ffmpeg

Models
- images/control-net: https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/main/v1-5-pruned-emaonly-fp16.safetensors
- video: https://huggingface.co/THUDM/CogVideoX-2b

# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
# Use StableDiffusionXLPipeline for SDXL models (juggernautXL_juggXIByRundiffusion.safetensors)
# Use StableDiffusionPipeline for SD 1.5 models (v1-5-pruned-emaonly-fp16.safetensors)

Read Metadata
from PIL import Image
img = Image.open("output.png")
metadata = img.info.get("RenderRequest")  # Returns JSON string

Debug
print("---------------------------------")
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(next(controlnet_pipeline.unet.parameters()).device)
print("---------------------------------")
"""

import time
import torch
import gc 
from diffusers import (
    StableDiffusionPipeline, StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline,
    ControlNetModel, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    CogVideoXPipeline)
from diffusers.utils import export_to_video
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json

MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors" # SD1.5
# MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors" # SDXL

CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/control_v11p_sd15_scribble_fp16.safetensors" # SD1.5
# CONTROLNET_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/controlnet/diffusion_pytorch_model_canny.fp16.safetensors" # SDXL

VIDEO_MODEL_PATH = "E:/Models/CogVideoX-2b"

width=512
height=512

# Lazy-loaded pipelines
current_pipeline = None
current_type = None
current_has_controlnet = None


def get_image_pipeline(with_controlnet: bool = False):
    """
    Load image pipeline, clearing video pipeline if loaded.
    If with_controlnet is True, use it to create the controlnet image.
    """

    global current_pipeline, current_type, current_has_controlnet

    # Reload if switching pipeline type OR if controlnet requirement changed
    if current_type != "image" or current_has_controlnet != with_controlnet:
        if current_pipeline:
            del current_pipeline
            gc.collect() # <--- ADDED: Force RAM cleanup
            torch.cuda.empty_cache()

        if with_controlnet:
            print(">> Loading controlnet pipeline")
            controlnet = ControlNetModel.from_single_file(CONTROLNET_PATH, torch_dtype=torch.float16)
            current_pipeline = StableDiffusionControlNetPipeline.from_single_file(
                MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16)
        else:  
            print(">> Loading image pipeline")
            current_pipeline = StableDiffusionPipeline.from_single_file(MODEL_PATH, torch_dtype=torch.float16)  

        current_pipeline.to("cuda")
        current_pipeline.enable_attention_slicing()
        current_pipeline.vae.enable_slicing()
        current_type = "image"
        current_has_controlnet = with_controlnet

    return current_pipeline


def get_video_pipeline():
    """
    Load video pipeline, clearing image pipeline if loaded
    """

    global current_pipeline, current_type, current_has_controlnet
    
    if current_type != "video":
        if current_pipeline:
            del current_pipeline
            gc.collect() # <--- ADDED: Force RAM cleanup
            torch.cuda.empty_cache()
        
        print(">> Loading CogVideoX pipeline")
        current_pipeline = CogVideoXPipeline.from_pretrained(VIDEO_MODEL_PATH, torch_dtype=torch.float16)

        # Optimizations for 6GB GPU
        current_pipeline.enable_sequential_cpu_offload()
        current_pipeline.vae.enable_slicing()
        current_pipeline.vae.enable_tiling()

        current_type = "video"
        current_has_controlnet = None
        
    return current_pipeline


def get_scheduler(pipe, sampler_name: str):
    """
    Set samlper based on sampler name
    """

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
    pipe = get_image_pipeline(bool(control_image_path))
    pipe.scheduler = get_scheduler(pipe, sampler)
    
    # Render with controlnet
    if control_image_path:
        control_image = Image.open(control_image_path).convert("RGB")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=control_image,
            num_inference_steps=steps,
            generator=generator,
            controlnet_conditioning_scale=controlnet_strength,
            guidance_scale=cfg,
            width=width,
            height=height).images[0]

    # Render without controlnet
    else:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=cfg,
            width=width,
            height=height).images[0]
    
    # Save image with metadata
    metadata = PngInfo()
    render_request = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "seed": seed,
        "cfg": cfg,
        "sampler": sampler,
        "controlnet_strength": controlnet_strength,
        "control_image_path": control_image_path
    }
    metadata.add_text("RenderRequest", json.dumps(render_request))
    image.save(output_path, pnginfo=metadata)
    
    # Measure render time and format as MM:SS
    render_time = time.perf_counter() - start_time
    minutes = int(render_time // 60)
    seconds = int(render_time % 60)
    return f"{minutes:02d}:{seconds:02d}"


def render_video(prompt: str, negative_prompt: str, steps: int, seed: int, cfg: float,
                 frames: int, fps: int, output_path: str):
    """
    Render video using CogVideoX model
    """

    start_time = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    pipe = get_video_pipeline()

    # Fixed resolution 720x480 required for CogVideoX-2b
    video = pipe(
    prompt=prompt,
    num_inference_steps=steps,
    guidance_scale=cfg,
    num_frames=frames,
    height=height,
    width=width,
    generator=generator
    ).frames[0]


    export_to_video(video, output_path, fps=fps)
    
    render_time = time.perf_counter() - start_time
    return f"{int(render_time // 60):02d}:{int(render_time % 60):02d}"