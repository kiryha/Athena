"""
FastAPI backend for Athena image generator
"""

import secrets
import threading
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import renderer

app = FastAPI()

IMAGES_DIR = Path("E:/images")
FRONTEND_DIR = Path(__file__).parent / "web" / "dist"


class ImageRenderRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    seed: int = 0
    cfg: float = 2.1
    sampler: str = "DPM++ 2M"
    control_image_path: str = ""
    controlnet_strength: float = 0.8


class VideoRenderRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    seed: int = 0
    cfg: float = 7.5
    frames: int = 24
    fps: int = 8


@app.post("/render-image")
def handle_image_render(req: ImageRenderRequest):

    # Define images paths
    filename = f"{secrets.token_hex(4).upper()}.png"
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
        control_image_path=req.control_image_path if req.control_image_path else None
    )

    return {"image_url": f"/images/{filename}", "image_path": str(output_path), "render_time": render_time}


@app.post("/render-video")
def handle_video_render(req: VideoRenderRequest):
    filename = f"{secrets.token_hex(4).upper()}.mp4"
    output_path = IMAGES_DIR / filename
    
    render_time = renderer.render_video(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        steps=req.steps,
        seed=req.seed,
        cfg=req.cfg,
        frames=req.frames,
        fps=req.fps,
        output_path=str(output_path)
    )
    
    return {"video_url": f"/images/{filename}", "video_path": str(output_path), "render_time": render_time}


@app.get("/pick-file")
def pick_file():
    """Open native file picker and return selected file path."""
    import tkinter as tk
    from tkinter import filedialog
    
    selected_path = []
    
    def run_dialog():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select ControlNet Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        selected_path.append(file_path)
        root.destroy()
    
    # Run tkinter in main thread
    thread = threading.Thread(target=run_dialog)
    thread.start()
    thread.join()
    
    return {"path": selected_path[0] if selected_path else ""}


# Serve generated images
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
# Serve React frontend (built files)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
