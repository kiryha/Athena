"""
FastAPI backend for Athena image generator
"""

import uuid
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from renderer import render

app = FastAPI()

IMAGES_DIR = Path("E:/images")


class RenderRequest(BaseModel):
    prompt: str
    steps: int = 30
    seed: int = 0


@app.post("/render")
def render_image(req: RenderRequest):
    filename = f"{uuid.uuid4()}.png"
    output_path = IMAGES_DIR / filename
    render(req.prompt, req.steps, req.seed, str(output_path))
    return {"image_url": f"/images/{filename}"}


# Serve generated images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Serve React frontend (built files)
FRONTEND_DIR = Path(__file__).parent / "web" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
