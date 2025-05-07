import base64
import os
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException

from src.presentation.fastapi.routes.upload.schemas import UploadResponse

ROUTER = APIRouter(prefix="/upload")


@ROUTER.post("")
async def upload_video(
        file: UploadFile = File(...),

):
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    if not file.filename.lower().endswith(tuple(ALLOWED_VIDEO_EXTENSIONS)):
        raise ValueError(
            "Файл не является поддерживаемым видеофайлом. Допустимые форматы: " + ", ".join(ALLOWED_VIDEO_EXTENSIONS))

    try:
        os.makedirs("video", exist_ok=True)
        file_name = f"{uuid.uuid4()}.mp4"
        project_root = os.environ.get('PYTHONPATH', '').split(os.pathsep)[0]
        project_root = Path(project_root)
        video_path = project_root / "video" / file_name

        with open(video_path, "wb") as f:
            content = await file.read()
            f.write(content)

        cap = cv2.VideoCapture(str(video_path))
        success, frame = cap.read()
        cap.release()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to extract the first frame from the video")

        _, buffer = cv2.imencode('.jpg', frame)
        first_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return UploadResponse(file_name=file_name, first_frame_base64=first_frame_base64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
