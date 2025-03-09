from typing import Optional

import cv2
import numpy as np

from src.application.enums.device import Device
from src.application.services.cv2.methods import show_frame_while_drawing_regions
from src.application.services.yolo.methods import process_video_two_region

class CountTwoRegion:
    def __init__(self, video_path: str, device: Device, headless: bool):
        self.video_path = video_path
        self.device = device
        self.headless = headless
        self.region_coordinates = []
        self.cars_count = {}

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
        self.region_coordinates = show_frame_while_drawing_regions(self.get_first_frame())

    def process_video(self):
        print([r_coordinates.__dict__ for r_coordinates in self.region_coordinates])
        self.cars_count = process_video_two_region.run(self.region_coordinates, source=self.video_path, device=self.device, headless=self.headless)

    def execute(self):
        self.get_region_coordinates()
        self.process_video()
        return self.cars_count

if __name__ == "__main__":
    print(CountTwoRegion("/Users/alexandrbelyanin/Documents/PycharmProjects/v54_good/video/5.mp4", Device.cpu, True).execute())

