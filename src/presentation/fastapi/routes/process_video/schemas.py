from typing import List

from pydantic import BaseModel

from src.application.schemas.region_coordinates import Coordinates


class ProcessVideoRequest(BaseModel):
    file_name: str
    start: List[Coordinates]
    end: List[Coordinates]


class ProcessVideoResponse(BaseModel):
    cars_cnt: int
