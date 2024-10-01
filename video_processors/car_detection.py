import cv2
import numpy as np
import os
from video_processors.utils import check_intersection, increase_brightness


class CarDetector:
    def __init__(self):
        self.cars = {}
        self.next_car_id = 0
        self.red = 0
        self.blue = 0
        self.green = 0
        self.black = 0
        self.load_yolo()

    def load_yolo(self):
        weights_path = os.path.abspath("ai_config/yolov3.weights")
        cfg_path = os.path.abspath("ai_config/yolov3.cfg")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов YOLO не найден: {weights_path}")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Файл конфигурации YOLO не найден: {cfg_path}")

        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        with open("ai_config/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect_cars(self, video_processor, regions, frame_height, frame_weight):
        net, classes = self.net, self.classes
        self.regions = regions
        self.set_mask(frame_height,frame_weight)

        while True:
            frame = video_processor.get_frame()
            if frame is None:
                break

            processed_frame = increase_brightness(frame)
            processed_frame = video_processor.preprocess_frame(processed_frame)
            cv2.imshow('Car Detector', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            detected_cars = self.detect_cars_yolo(processed_frame, net)
            frame_with_cars = self.draw_cars(detected_cars,processed_frame)
            cv2.imshow('Cars Detection', frame_with_cars)

        cv2.destroyAllWindows()
        tr = open('result/cars_info.txt', 'w', encoding='utf-8')
        string = f"""
            RED: {self.red};
            GREEN: {self.green};
            BLUE: {self.blue};
            BLACK: {self.black};
            """
        tr.write(string)

    def detect_cars_yolo(self, frame, net):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(self.get_output_layers(net))
        detected_cars = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 2 and confidence > 0.5:  # Class ID for 'car' in COCO dataset
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    detected_cars.append((x, y, w, h, center_x, center_y, confidence))
        return detected_cars

    def set_mask(self, frame_height, frame_weight):
        self.red_mask = np.zeros((frame_height, frame_weight), dtype=np.uint8)
        cv2.fillPoly(self.red_mask, np.array([self.regions["red_points"]], dtype=np.int32), 255)
        self.blue_mask = np.zeros((frame_height, frame_weight), dtype=np.uint8)
        cv2.fillPoly(self.blue_mask, np.array([self.regions["blue_points"]], dtype=np.int32), 255)
        self.green_mask = np.zeros((frame_height, frame_weight), dtype=np.uint8)
        cv2.fillPoly(self.green_mask, np.array([self.regions["green_points"]], dtype=np.int32), 255)
        self.black_mask = np.zeros((frame_height, frame_weight), dtype=np.uint8)
        cv2.fillPoly(self.black_mask, np.array([self.regions["black_points"]], dtype=np.int32), 255)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_cars(self, detected_cars, processed_frame):
        new_cars = {}
        for (x, y, w, h, center_x, center_y, confidence) in detected_cars:
            for existing_car in detected_cars:
                existing_x, existing_y, existing_w, existing_h, _, _, _ = existing_car
                if check_intersection((x, y, w, h), (existing_x, existing_y, existing_w, existing_h)):
                    detected_cars.remove(existing_car)
            car_img = processed_frame[y:y + h, x:x + w]
            car_id = None

            if self.red_mask[center_y, center_x] == 255:
                region = 'red'
                color = (0, 0, 255)
            elif self.blue_mask[center_y, center_x] == 255:
                region = 'blue'
                color = (255, 0, 0)
            elif self.green_mask[center_y, center_x] == 255:
                region = 'green'
                color = (0, 255, 0)
            elif self.black_mask[center_y, center_x] == 255:
                region = 'black'
                color = (0, 0, 0)
            else:
                region = None
                color = (255, 255, 255)
            # intersect = False

            if region:
                found = False
                for id, data in self.cars.items():
                    (prev_x, prev_y, prev_w, prev_h, prev_center_x, prev_center_y, prev_region, prev_color) = data
                    if check_intersection((x, y, w, h), (prev_x, prev_y, prev_w, prev_h)):
                        car_id = id
                        found = True
                        break

                if not found:
                    car_id = self.next_car_id
                    self.next_car_id += 1

                new_cars[car_id] = (x, y, w, h, center_x, center_y, region, color)

                if car_id not in self.cars:
                    if region == 'red':
                        self.red += 1
                    elif region == 'blue':
                        self.blue += 1
                    elif region == 'green':
                        self.green += 1
                    elif region == 'black':
                        self.black += 1

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)

        self.cars = new_cars

        return processed_frame
