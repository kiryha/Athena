"""
Hello World AI image generation

# To uninstall: pip uninstall -y torch torchvision torchaudio xformers

# Prerequisites:
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

"""

import torch
from diffusers import StableDiffusionXLPipeline

model_path = "E:/Projects/ComfyUI_windows_portable/ComfyUI/models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors"
prompt = "photorealistic robot with yellow painted metal"
output_image = "E:/robot_001.png"


pipeline = StableDiffusionXLPipeline.from_single_file(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipeline.to("cuda")
pipeline.enable_attention_slicing()
# pipeline.enable_xformers_memory_efficient_attention()

image = pipeline(prompt, num_inference_steps=20).images[0]
image.save(output_image)