"""
Hello World AI image generation

# To uninstall: pip uninstall -y torch torchvision torchaudio xformers

# Prerequisites:
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# RUN THIS IN CMD: before running the script!!!
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
"""

import torch
from diffusers import StableDiffusionPipeline


# model_path = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
model_path = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"
prompt = "photorealistic robot with yellow painted metal"
output_image = "E:/image_001.png"

pipe = StableDiffusionPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.to("cuda")
torch_generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(prompt,  num_inference_steps=5, generator=torch_generator).images[0]
image.save(output_image)