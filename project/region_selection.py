import cv2
import numpy as np

class RegionSelector:
    def __init__(self):
        self.red_points = []
        self.blue_points = []
        self.green_points = []
        self.black_points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.red_points) < 4:
                self.red_points.append((x, y))
            elif len(self.blue_points) < 4:
                self.blue_points.append((x, y))
            elif len(self.green_points) < 4:
                self.green_points.append((x, y))
            else:
                self.black_points.append((x, y))

    def select_regions(self, frame):
        cv2.namedWindow('Select Regions')
        cv2.setMouseCallback('Select Regions', self.mouse_callback)

        while True:
            frame_copy = frame.copy()
            self.draw_regions(frame_copy, self.red_points, (0, 0, 255))
            self.draw_regions(frame_copy, self.blue_points, (255, 0, 0))
            self.draw_regions(frame_copy, self.green_points, (0, 255, 0))
            self.draw_regions(frame_copy, self.black_points, (0, 0, 0))

            cv2.imshow('Select Regions', frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Select Regions')

    def draw_regions(self, image, points, color):
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(image, points[i - 1], points[i], color, 2)
            cv2.line(image, points[-1], points[0], color, 2)

    def validate_regions(self):
        return len(self.red_points) >= 4 and len(self.blue_points) >= 4 and len(self.green_points) >= 4 and len(self.black_points) >= 4
