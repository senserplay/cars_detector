from typing import List

from pydantic import BaseModel


class Coordinates(BaseModel):
    x: int
    y: int


class RegionCoordinates(BaseModel):
    region_id: int
    region_name: str
    vertices: List[Coordinates]
