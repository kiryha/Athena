"""
FastAPI backend for Athena image generator
"""

import secrets
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from renderer import render_image

app = FastAPI()

IMAGES_DIR = Path("E:/images")
FRONTEND_DIR = Path(__file__).parent / "web" / "dist"


class RenderRequest(BaseModel):
    prompt: str
    steps: int = 30
    seed: int = 0


@app.post("/render")
def handle_render(req: RenderRequest):

    control_image_path = "C:/Users/kko8/OneDrive/projects/houdini_snippets/prod/3d/render/athena/ctr_images/05K_alehandro_canny.jpg"
    filename = f"{secrets.token_urlsafe(6).upper()}.png"
    output_path = IMAGES_DIR / filename

    render_time = render_image(req.prompt, req.steps, req.seed, str(output_path), control_image_path)

    return {"image_url": f"/images/{filename}", "image_path": str(output_path), "render_time": render_time}


# Serve generated images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
# Serve React frontend (built files)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
