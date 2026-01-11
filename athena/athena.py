"""
FastAPI backend for Athena image generator
"""

import secrets
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import renderer

app = FastAPI()

IMAGES_DIR = Path("E:/images")
FRONTEND_DIR = Path(__file__).parent / "web" / "dist"


class RenderRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    seed: int = 0
    cfg: float = 2.1
    sampler: str = "DPM++ 2M"
    controlnet_strength: float = 0.8


@app.post("/render")
def handle_render(req: RenderRequest):

    # Definae images paths
    control_image_path = "C:/Users/kko8/OneDrive/projects/houdini_snippets/prod/3d/render/athena/ctr_images/SDXL_alehandro_canny.jpg"
    filename = f"{secrets.token_urlsafe(6).upper()}.png"
    output_path = IMAGES_DIR / filename

    # Run render
    render_time = renderer.render_image(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        steps=req.steps,
        seed=req.seed,
        cfg=req.cfg,
        sampler=req.sampler,
        controlnet_strength=req.controlnet_strength,
        output_path=str(output_path),
        control_image_path=control_image_path
    )

    return {"image_url": f"/images/{filename}", "image_path": str(output_path), "render_time": render_time}


# Serve generated images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
# Serve React frontend (built files)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
