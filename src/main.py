from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.presentation.fastapi.setup_routes import setup_routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_routes(app)
