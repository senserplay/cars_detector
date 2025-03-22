from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi import Form
import json
from src.application.enums.device import Device
from src.application.schemas.region_coordinates import Coordinates, RegionCoordinates
from src.usecase.cars.count_two_region import CountTwoRegion
from typing import Dict, List, Tuple


app = FastAPI()
templates = Jinja2Templates(directory="src/templates")

class VideoRequest(BaseModel):
    file_name: str
    regions: Dict[str, Dict[str, List[Tuple[float, float]]]]
    device_name: str
    headless_mode: bool

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/uploadfile")
async def upload_video(
    file: UploadFile = File(...),
    data: str = Form(...)
):
    try:
        data_dict = json.loads(data)
        video_request = VideoRequest(**data_dict)
        file_name = video_request.file_name
        device_name = Device[video_request.device_name]
        haedless_mode = video_request.headless_mode

        #На это я пока забил, если этот код вообще нужен будет
        file_content = await file.read()

        region_coordinates = []
        for i, (region_name, region_data) in enumerate(video_request.regions.items(), start=1):
            region_coordinates.append(
                RegionCoordinates(
                    region_id=i,
                    region_name=region_name,
                    vertices=[Coordinates(x=point[0], y=point[1]) for point in region_data["points"]]
                )
            )

        result = CountTwoRegion(file_name, device_name, haedless_mode, region_coordinates).execute()

        return {
            "cars_count": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#запусти в терминале uvicorn src.presentation.fastapi.api:app --reload
