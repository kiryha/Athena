"""
Hello World AI image generation
"""

import torch
from diffusers import StableDiffusionXLPipeline


model_path = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
prompt = "photorealistic robot with yellow painted metal"
output_image = "E:/robot_001.png"


pipeline = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16
)

pipeline.to("cuda")

image = pipeline(prompt).images[0]
image.save(output_image)