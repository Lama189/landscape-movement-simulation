import asyncio
from typing import Optional
from fastapi import APIRouter, Request, Form, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from landscape_generator import LandscapeGenerator, plot_topview
from shared.presets import PRESETS

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/run_pathfinding", response_class=HTMLResponse)
async def run_pathfinding(request: Request, points: list[dict] = Body(...)):
    params = request.session.get("generator_params")
    if not params:
        return HTMLResponse("Ошибка: параметры генератора не найдены. Сначала сгенерируйте ландшафт.", status_code=400)

    start = (points[0]["x"], points[0]["y"], 0, 0)
    goal = (points[1]["x"], points[1]["y"])

    generator = LandscapeGenerator(**params)
    xs, ys, zs, df = generator.find_path_discrete(start, goal)

    fig_html = generator.plot(return_html=True, with_path=True, animate=True, start=start, goal=goal)
    table_html = df.to_html(classes="table table-striped", index=False)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "fig_html": fig_html,
            "table_html": table_html,
            "has_path": not df.empty,
        },
    )


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

    request.session["generator_params"] = {
        "width": generator.width,
        "height": generator.height,
        "scale": generator.scale,
        "grid_size": generator.grid_size,
        "zscale": generator.zscale,
        "seed": generator.seed,
    }

    height_map = generator.generate_heightmap()
    fig_html = plot_topview(height_map, return_html=True)

    return templates.TemplateResponse(
        "select_points.html",
        {
            "request": request,
            "fig_html": fig_html,
        },
    )



