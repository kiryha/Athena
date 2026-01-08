"""
Image renderer using Stable Diffusion
"""

import torch
from diffusers import StableDiffusionPipeline

MODEL_PATH = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"

pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")


def render_image(prompt: str, steps: int, seed: int, output_path: str):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt, num_inference_steps=steps, generator=generator).images[0]
    image.save(output_path)
