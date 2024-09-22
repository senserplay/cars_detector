import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, buffer_size=10):
        self.cap = None
        self.frame_buffer = []
        self.buffer_size = buffer_size

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def preprocess_frame(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(gray_frame)
        return cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)

    def remove_static_background(self, frame):
        processed_frame = self.preprocess_frame(frame)

        if len(self.frame_buffer) < self.buffer_size:
            self.frame_buffer.append(processed_frame)
            return processed_frame

        self.frame_buffer.pop(0)
        self.frame_buffer.append(processed_frame)

        difference = cv2.absdiff(processed_frame, self.frame_buffer[0])
        for prev_frame in self.frame_buffer[1:]:
            difference = cv2.bitwise_or(difference, cv2.absdiff(processed_frame, prev_frame))

        kernel = np.ones((5, 5), np.uint8)
        difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

        _, mask = cv2.threshold(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(processed_frame, processed_frame, mask=mask)

    def release(self):
        self.cap.release()
