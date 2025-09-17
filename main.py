from backend.common.handlers import router as common_router
from fastapi import FastAPI

app = FastAPI()

app.include_router(common_router)