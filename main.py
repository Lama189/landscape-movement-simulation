from backend.common.handlers import router as common_router
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="SUPER_SECRET_KEY")
app.include_router(common_router)