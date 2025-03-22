from fastapi import FastAPI
from src.presentation.fastapi.routes.process_video.api import ROUTER as PROCESS_VIDEO_ROUTER


def setup_routes(app: FastAPI):
    app.include_router(PROCESS_VIDEO_ROUTER)