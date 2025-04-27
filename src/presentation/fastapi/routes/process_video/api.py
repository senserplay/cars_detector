from fastapi import APIRouter, HTTPException

from src.application.enums.device import Device
from src.application.schemas.region_coordinates import RegionCoordinates
from src.presentation.fastapi.routes.process_video.schemas import ProcessVideoRequest, ProcessVideoResponse
from src.usecase.cars.count_two_region import CountTwoRegion

ROUTER = APIRouter(prefix="/process-video")

@ROUTER.post("")
async def process_video(
        request: ProcessVideoRequest

):
    try:
        start = RegionCoordinates(region_id=1, region_name="start", vertices=request.start)
        end = RegionCoordinates(region_id=2, region_name="end", vertices=request.end)
        regions = [start, end]

        cars_cnt = CountTwoRegion(request.file_name, Device.cuda,True, regions).execute()
        return ProcessVideoResponse(cars_cnt=cars_cnt)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=str(e)
        )