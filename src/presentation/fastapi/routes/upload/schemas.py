from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_name: str
    first_frame_base64: str