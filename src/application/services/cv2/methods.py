import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

from src.application.enums.device import Device
from src.application.schemas.region_coordinates import RegionCoordinates, Coordinates


from typing import List, Optional
from shapely.geometry import Polygon, Point
import cv2


# Глобальные переменные
created_regions = []  # Хранит созданные регионы
current_zone_points = []  # Точки для текущего региона
color_index = 0  # Индекс цвета для регионов
zone_colors = [(255, 0, 0), (0, 255, 0)]  # Цвета для регионов
current_region: Optional[dict] = None  # Текущий перетаскиваемый регион


def mouse_callback(event, x, y, flags, param):
    global created_regions, current_zone_points, color_index, current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(created_regions) < 2:  # Создаем только два региона
            current_zone_points.append((x, y))  # Добавляем точку

            # Если установлено 4 точки, создаём четырехугольную зону
            if len(current_zone_points) == 4:
                # Преобразуем точки в формат Coordinates
                vertices = [Coordinates(x=p[0], y=p[1]) for p in current_zone_points]

                # Создаем новый регион
                region_name = "start" if len(created_regions) == 0 else "end"
                created_regions.append({
                    "name": region_name,
                    "vertices": vertices,  # Вершины четырехугольника
                    "dragging": False,
                    "region_color": zone_colors[color_index % len(zone_colors)],
                    "text_color": (255, 255, 255),
                    "counts": 0,
                    "tracked_ids": set(),
                })

                current_zone_points = []  # Сбрасываем текущие точки
                color_index += 1  # Переходим к следующему цвету

        # Проверяем, находится ли курсор внутри какого-либо региона
        for region in created_regions:
            # Проверяем, находится ли точка внутри многоугольника
            contour = np.array([[v.x, v.y] for v in region["vertices"]])
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x - region["vertices"][0].x
                current_region["offset_y"] = y - region["vertices"][0].y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            # Вычисляем смещение
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]

            # Обновляем координаты всех вершин региона
            for vertex in current_region["vertices"]:
                vertex.x += dx - vertex.x
                vertex.y += dy - vertex.y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def draw_regions(frame, regions: List[RegionCoordinates], color=(255, 0, 0), thickness=2, font_scale=0.7,
                 text_color=(255, 255, 255)):
    """
    Рисует регионы на кадре и добавляет название региона в его центр.

    :param frame: Кадр из видео (numpy array)
    :param regions: Список объектов RegionCoordinates
    :param color: Цвет границы региона (BGR формат)
    :param thickness: Толщина линии границы региона
    :param font_scale: Размер шрифта для названия региона
    :param text_color: Цвет текста названия региона (BGR формат)
    """
    for region in regions:
        # Получаем вершины региона
        vertices = [(v.x, v.y) for v in region.vertices]

        # Преобразуем вершины в массив NumPy
        contour = np.array(vertices, dtype=np.int32)

        # Рисуем контур региона
        cv2.polylines(frame, [contour], isClosed=True, color=color, thickness=thickness)

        # Вычисляем центр региона
        moments = cv2.moments(contour)
        if moments["m00"] != 0:  # Проверяем, чтобы избежать деления на ноль
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            center_x, center_y = contour.mean(axis=0).astype(int)

        # Добавляем текст с названием региона в центр
        text_size, _ = cv2.getTextSize(region.region_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_width, text_height = text_size
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Рисуем фон для текста
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            color,
            -1  # Заполняем прямоугольник полностью
        )

        # Рисуем сам текст
        cv2.putText(
            frame,
            region.region_name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2,
            cv2.LINE_AA
        )

def show_frame_while_drawing_regions(frame):
    """
    Отображает кадр и позволяет пользователю рисовать до двух четырехугольных регионов.
    :param frame: Кадр, на котором будут рисоваться регионы.
    :return: Список созданных регионов в формате RegionCoordinates.
    """
    global created_regions, current_zone_points, color_index, current_region

    # Создаем окно для рисования регионов
    window_name = "Draw Regions"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        frame_copy = frame.copy()

        # Рисуем созданные регионы
        for region in created_regions:
            # Извлекаем вершины многоугольника
            points = np.array([[v.x, v.y] for v in region["vertices"]])

            # Рисуем контур
            cv2.polylines(
                frame_copy,
                [points.reshape((-1, 1, 2))],
                isClosed=True,
                color=region["region_color"],
                thickness=2
            )

            # Рисуем название региона
            text_position = points.mean(axis=0).astype(int)  # Центр многоугольника
            cv2.putText(
                frame_copy,
                region["name"],
                tuple(text_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                region["text_color"],
                2
            )

        # Отображаем кадр
        cv2.imshow(window_name, frame_copy)

        # Обработка нажатий клавиш
        key = cv2.waitKey(1)
        if key == ord('q'):  # Нажмите 'q' для выхода
            break
        elif key == ord('c'):  # Нажмите 'c' для очистки регионов
            created_regions = []
            current_zone_points = []
            color_index = 0
            current_region = None

        # Прерываем цикл, если созданы два региона
        if len(created_regions) == 2:
            break

    cv2.destroyAllWindows()
    return get_regions_as_coordinates()


def get_regions_as_coordinates() -> List[RegionCoordinates]:
    """
    Возвращает созданные регионы в формате RegionCoordinates.
    """
    regions = []
    for idx, region in enumerate(created_regions):
        regions.append(
            RegionCoordinates(
                region_id=idx + 1,
                region_name=region["name"],
                vertices=region["vertices"]
            )
        )
    return regions


# Пример использования
if __name__ == "__main__":
    # Создаем тестовый кадр
    frame = np.zeros((800, 800, 3), dtype=np.uint8)

    # Запускаем функцию для рисования регионов
    regions = show_frame_while_drawing_regions(frame)

    # Выводим созданные регионы
    for region in regions:
        print(f"Region {region.region_id}: {region.region_name}, "
              f"Start: ({region.start.x}, {region.start.y}), "
              f"End: ({region.end.x}, {region.end.y})")