# app.py
from fastapi import FastAPI
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from animatediff import AnimateDiffPipeline
import torch
from io import BytesIO
import base64

app = FastAPI()

# Глобальные переменные моделей
sdxl_pipe = None
anim_pipe = None
flux_pipe = None

@app.on_event("startup")
def load_models():
    global sdxl_pipe, anim_pipe, flux_pipe

    # 1. Flux Schnell
    print("📥 Загрузка Flux Schnell...")
    flux_pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    flux_pipe.to("cuda")

    # 2. SDXL
    print("📥 Загрузка SDXL...")
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    )
    sdxl_pipe.to("cuda")

    # 3. AnimateDiff + SDXL
    print("📥 Загрузка AnimateDiff...")
    anim_pipe = AnimateDiffPipeline.from_pretrained(
        sdxl_pipe,
        "guoyww/animatediff-motion-module-sdxl"
    )
    anim_pipe.to("cuda")

    print("✅ Все модели загружены!")

# Эндпоинты
@app.get("/generate/flux")
async def generate_flux(prompt: str):
    image = flux_pipe(prompt, num_inference_steps=4).images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "photo", "data": img_str}

@app.get("/generate/sdxl")
async def generate_sdxl(prompt: str):
    image = sdxl_pipe(prompt).images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "photo", "data": img_str}

@app.get("/generate/anim")
async def generate_anim(prompt: str):
    result = anim_pipe(prompt, num_frames=16)
    buffer = BytesIO()
    result.save(buffer, format="GIF")
    gif_str = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "video", "data": gif_str}