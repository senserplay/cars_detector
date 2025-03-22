import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import TypeAdapter

from src.application.enums.device import Device
from src.application.schemas.region_coordinates import Coordinates, RegionCoordinates
from src.usecase.cars.count_two_region import CountTwoRegion

ROUTER = APIRouter(prefix="/process-video")

@ROUTER.post("")
async def process_video(
        file: UploadFile = File(...),
        start: str = Form(...),
        end: str = Form(...)

):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="File must be in MP4 format")
    try:
        start_coords = TypeAdapter(List[Coordinates]).validate_json(start)
        end_coords = TypeAdapter(List[Coordinates]).validate_json(end)
        start = RegionCoordinates(region_id=1, region_name="start", vertices=start_coords)
        end = RegionCoordinates(region_id=2, region_name="end", vertices=end_coords)
        regions = [start, end]
    except:
        raise HTTPException(status_code=400, detail="Invalid coordinates format")

    try:
        project_root = os.environ.get('PYTHONPATH', '').split(os.pathsep)[0]
        project_root = Path(project_root)
        video_path = project_root / "video" / file.filename

        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    cnt_cars = CountTwoRegion(file.filename,Device.cuda,True, regions).execute()
    return {
        "cnt_cars": cnt_cars,
    }