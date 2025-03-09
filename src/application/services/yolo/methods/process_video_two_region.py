import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from src.application.enums.device import Device
from src.application.schemas.region_coordinates import RegionCoordinates, Coordinates
from src.application.services.cv2.methods import draw_regions
track_history = defaultdict(list)


def on_car_detected(
    regions: List[RegionCoordinates],
    bbox,
    car_id,
    center,
    counted_cars: dict
):
    """
    Эта функция вызывается, когда обнаружена машина.
    :param regions: Список регионов (четырехугольников)
    :param bbox: Координаты ограничивающего прямоугольника (x1, y1, x2, y2)
    :param car_id: Уникальный ID машины
    :param center: Центр объекта (x_center, y_center)
    :param counted_cars: Словарь для отслеживания машин в регионах
    :return: Обновленный словарь counted_cars
    """
    start_region = None
    end_region = None

    # Определяем стартовый и конечный регионы
    for region in regions:
        if region.region_name == "start":  # Предполагаем, что первый регион — стартовый
            start_region = region
        elif region.region_name == "end":  # Второй регион — конечный
            end_region = region

    # Если стартовый или конечный регион не определен, выходим
    if not start_region or not end_region:
        return counted_cars

    # Преобразуем вершины регионов в массив numpy для использования cv2.pointPolygonTest
    start_contour = np.array([[v.x, v.y] for v in start_region.vertices])
    end_contour = np.array([[v.x, v.y] for v in end_region.vertices])

    # Проверяем, находится ли центр машины внутри стартового региона
    if car_id not in counted_cars:
        if cv2.pointPolygonTest(start_contour, center, False) >= 0:  # Точка внутри многоугольника
            counted_cars[car_id] = {"start": True, "end": False}

    # Проверяем, находится ли центр машины внутри конечного региона
    elif car_id in counted_cars and not counted_cars[car_id]["end"]:
        if cv2.pointPolygonTest(end_contour, center, False) >= 0:  # Точка внутри многоугольника
            counted_cars[car_id]["end"] = True
    return counted_cars


def run(
        regions: List[RegionCoordinates],
        headless: bool = True,
        weights: str = "yolo11n.pt",
        source: str = None,
        device: Device = Device.cpu,
        view_img: bool = False,
        classes=None,
        line_thickness: int = 2,
        track_thickness: int = 2,
        region_thickness: int = 2,
):
    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    project_root = os.environ.get('PYTHONPATH', '').split(os.pathsep)[0]
    project_root = Path(project_root)
    weights_path = project_root / "src" / "application" / "services" / "yolo" / "weights" / weights

    # Setup Model
    model = YOLO(str(weights_path))
    model.to(device.value)
    names = model.model.names

    cap = cv2.VideoCapture(source)

    paused = False
    counted_cars = {}
    while True:
        if not paused:
            ret, frame = cap.read()
            vid_frame_count += 1
            if vid_frame_count % 3 != 0:
                continue

            if not ret:
                print("Конец видео.")
                break

            results = model.track(frame, persist=True, classes=classes, conf=0.01)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)

                    counted_cars = on_car_detected(regions, (x1, y1, x2, y2), track_id, center, counted_cars)
                    if not headless:
                        # Рисуем bounding box и ID трека
                        annotator.box_label(box, f"{str(names[cls])} {track_id}", color=(0, 255, 0))

                    # Добавляем историю трекинга
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(center)
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                    # Рисуем линии трекинга
                    points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=track_thickness)
            if not headless:
                draw_regions(frame, regions)
                # Показываем кадр с обнаруженными объектами
                cv2.imshow("Video with detections", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Нажатие пробела для паузы
            paused = not paused  # Переключение флага паузы

    return sum([int(counted_cars[car]["start"] and counted_cars[car]["end"]) for car in counted_cars])
