import numpy as np
import cv2


def check_intersection(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)


def increase_brightness(image, value=30):
    return np.clip(image.astype(int) + value, 0, 255).astype(np.uint8)
