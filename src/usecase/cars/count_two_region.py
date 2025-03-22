import os
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from src.application.enums.device import Device
from src.application.schemas.region_coordinates import RegionCoordinates, Coordinates
from src.application.services.cv2.methods import show_frame_while_drawing_regions
from src.application.services.yolo.methods import process_video_two_region


class CountTwoRegion:
    def __init__(self, video_path: str, device: Device, headless: bool,
                 region_coordinates: List[RegionCoordinates] = []):
        self.video_path = video_path
        self.device = device
        self.headless = headless
        self.region_coordinates = region_coordinates
        self.cars_count = {}

    def prepare_video_path(self):
        project_root = os.environ.get('PYTHONPATH', '').split(os.pathsep)[0]
        project_root = Path(project_root)
        self.video_path = project_root / "video" / self.video_path

    def get_first_frame(self) -> Optional[np.ndarray]:
        """
        Извлекает первый кадр из видео.
        :return: Первый кадр в формате numpy array или None, если видео не удалось открыть.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Ошибка! Не удалось открыть видеофайл: {self.video_path}")
            return None

        ret, frame = cap.read()
        cap.release()  # Освобождаем ресурсы
        if ret:
            return frame
        else:
            print("Не удалось прочитать первый кадр.")
            return None

    def get_region_coordinates(self):
        if not self.region_coordinates:
            self.region_coordinates = show_frame_while_drawing_regions(self.get_first_frame())

    def process_video(self):
        print([r_coordinates.__dict__ for r_coordinates in self.region_coordinates])
        self.cars_count = process_video_two_region.run(self.region_coordinates, source=self.video_path,
                                                       device=self.device, headless=self.headless)

    def execute(self):
        self.prepare_video_path()
        self.get_region_coordinates()
        self.process_video()
        return self.cars_count