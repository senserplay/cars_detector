from fastapi import FastAPI
from src.presentation.fastapi.routes.process_video.api import ROUTER as PROCESS_VIDEO_ROUTER
from src.presentation.fastapi.routes.upload.api import ROUTER as UPLOAD_ROUTER


def setup_routes(app: FastAPI):
    app.include_router(PROCESS_VIDEO_ROUTER)
    app.include_router(UPLOAD_ROUTER)