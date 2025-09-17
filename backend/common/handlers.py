import asyncio
from typing import Optional
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from landscape_generator import LandscapeGenerator 
from shared.presets import PRESETS

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/", response_class=HTMLResponse)
async def generate_landscape(
    request: Request,
    width: int = Form(200),
    height: int = Form(100),
    scale: float = Form(20.0),
    preset: str = Form(""),
    seed: Optional[int] = Form(None),
    random_seed: str = Form(None),
):

    if random_seed:
        seed = None

    seed = int(seed) if seed not in (None, "", "null") else None
    generator = LandscapeGenerator(width=width, height=height, scale=scale, seed=seed)

    if preset and preset in PRESETS:
        params = PRESETS[preset]
        generator.scale = params["scale"]
        generator.grid_size = params["grid_size"]
        generator.zscale = params["zscale"]

    loop = asyncio.get_event_loop()
    fig_html = await loop.run_in_executor(
        None, lambda: generator.plot(return_html=True, with_path=True, animate=True)
    )

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "fig_html": fig_html, "preset": preset, "seed": seed},
    )
