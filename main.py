import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim

# Global variables
red_points, blue_points, green_points, black_points = [], [], [], []
car_images = []
red, blue, green, black = 0, 0, 0, 0
cars = {}
next_car_id = 0
frame_buffer = []
buffer_size = 10


def draw_regions(image, points, color):
    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color, 2)
        cv2.line(image, points[-1], points[0], color, 2)  # Close the polygon


def remove_static_background(frame):
    global frame_buffer

    # Применение предобработки кадра
    processed_frame = preprocess_frame(frame)

    # Если буфер ещё не заполнен, добавляем текущий кадр в буфер
    if len(frame_buffer) < buffer_size:
        frame_buffer.append(processed_frame)
        return processed_frame

    # Удаляем самый старый кадр из буфера и добавляем новый кадр
    frame_buffer.pop(0)
    frame_buffer.append(processed_frame)

    # Вычисляем разницу между текущим кадром и кадрами в буфере
    difference = cv2.absdiff(processed_frame, frame_buffer[0])
    for prev_frame in frame_buffer[1:]:
        difference = cv2.bitwise_or(difference, cv2.absdiff(processed_frame, prev_frame))

    # Применяем морфологическое преобразование для уменьшения шума
    kernel = np.ones((5, 5), np.uint8)
    difference = cv2.morphologyEx(difference, cv2.MORPH_OPEN, kernel)

    # Применяем бинаризацию для выделения областей с изменениями
    _, mask = cv2.threshold(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)

    # Применяем маску к текущему кадру для удаления статичного фона
    result = cv2.bitwise_and(processed_frame, processed_frame, mask=mask)

    return result


def mouse_callback(event, x, y, flags, param):
    global red_points, blue_points, green_points, black_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(red_points) < 4:
            red_points.append((x, y))
        elif len(blue_points) < 4:
            blue_points.append((x, y))
        elif len(green_points) < 4:
            green_points.append((x, y))
        else:
            black_points.append((x, y))


def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def detect_cars_yolo(frame, net, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
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


def find_most_similar_car(car_images, new_car_img):
    most_similar_img = None
    max_ssim = -1

    for car_img_str in car_images:
        car_img_str = cv2.resize(car_img_str, (128, 128))
        new_car_img = cv2.resize(new_car_img, (128, 128))

        new_gray = cv2.cvtColor(new_car_img, cv2.COLOR_BGR2GRAY)
        saved_gray = cv2.cvtColor(car_img_str, cv2.COLOR_BGR2GRAY)

        similarity = ssim(new_gray, saved_gray)

        if similarity > max_ssim:
            max_ssim = similarity
            most_similar_img = car_img_str

    return most_similar_img, max_ssim


def increase_brightness(image, value=30):
    # Увеличение яркости изображения путем добавления значения ко всем пикселям
    # value - значение, на которое увеличивается яркость (по умолчанию 30)
    return np.clip(image.astype(int) + value, 0, 255).astype(np.uint8)


# Пример использования:


def check_intersection(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)


def preprocess_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
    clahe_frame = clahe.apply(gray_frame)

    # Convert CLAHE output back to BGR
    enhanced_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_GRAY2BGR)

    return enhanced_frame


def main():
    global red_points, blue_points, green_points, black_points
    global red, blue, green, black
    global car_images, cars, next_car_id

    # Load YOLO model and classes
    net, classes = load_yolo()
    print('Укажите путь до видео')
    path = input()
    # Video capture
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow('Select Regions')
    cv2.setMouseCallback('Select Regions', mouse_callback)

    # Main loop for region selection
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # time.sleep(0.1)
        frame_with_regions = frame.copy()
        frame_with_regions = increase_brightness(frame_with_regions)
        frame_with_regions = preprocess_frame(frame_with_regions)
        draw_regions(frame_with_regions, red_points, (0, 0, 255))
        draw_regions(frame_with_regions, blue_points, (255, 0, 0))
        draw_regions(frame_with_regions, green_points, (0, 255, 0))
        draw_regions(frame_with_regions, black_points, (0, 0, 0))

        cv2.imshow('Select Regions', frame_with_regions)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow('Select Regions')

    if len(red_points) < 4 or len(blue_points) < 4 or len(green_points) < 4 or len(black_points) < 4:
        print("Error: All regions must be defined.")
        return

    # Create masks for each zone
    red_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(red_mask, np.array([red_points], dtype=np.int32), 255)
    blue_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(blue_mask, np.array([blue_points], dtype=np.int32), 255)
    green_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(green_mask, np.array([green_points], dtype=np.int32), 255)
    black_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(black_mask, np.array([black_points], dtype=np.int32), 255)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (127, 127, 0)
    thickness = 2

    # Main loop for car detection and counting
    i1 = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i1 % 10 != 0:
            i1 += 1
            continue
        i1 += 1
        hsv_frame = frame.copy()
        # hsv_frame = remove_static_background(frame)

        hsv_frame = increase_brightness(hsv_frame)
        hsv_frame = preprocess_frame(hsv_frame)

        detected_cars = detect_cars_yolo(hsv_frame, net, classes)

        # Update car tracking
        new_cars = {}
        for (x, y, w, h, center_x, center_y, confidence) in detected_cars:
            for existing_car in detected_cars:
                existing_x, existing_y, existing_w, existing_h, _, _, _ = existing_car
                if check_intersection((x, y, w, h), (existing_x, existing_y, existing_w, existing_h)):
                    detected_cars.remove(existing_car)
            car_img = hsv_frame[y:y + h, x:x + w]
            car_id = None

            if red_mask[center_y, center_x] == 255:
                region = 'red'
                color = (0, 0, 255)
            elif blue_mask[center_y, center_x] == 255:
                region = 'blue'
                color = (255, 0, 0)
            elif green_mask[center_y, center_x] == 255:
                region = 'green'
                color = (0, 255, 0)
            elif black_mask[center_y, center_x] == 255:
                region = 'black'
                color = (0, 0, 0)
            else:
                region = None
                color = (255, 255, 255)
            # intersect = False

            if region:
                found = False
                for id, data in cars.items():
                    (prev_x, prev_y, prev_w, prev_h, prev_center_x, prev_center_y, prev_region, prev_color) = data
                    if check_intersection((x, y, w, h), (prev_x, prev_y, prev_w, prev_h)):
                        car_id = id
                        found = True
                        break

                if not found:
                    car_id = next_car_id
                    next_car_id += 1

                new_cars[car_id] = (x, y, w, h, center_x, center_y, region, color)

                if car_id not in cars:
                    if region == 'red':
                        red += 1
                    elif region == 'blue':
                        blue += 1
                    elif region == 'green':
                        green += 1
                    elif region == 'black':
                        black += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cars = new_cars

        # Display counts
        image_with_text = cv2.putText(frame, f'Red: {red}', red_points[0], font, font_scale, text_color, thickness,
                                      cv2.LINE_AA)
        image_with_text = cv2.putText(frame, f'Blue: {blue}', blue_points[0], font, font_scale, text_color, thickness,
                                      cv2.LINE_AA)
        image_with_text = cv2.putText(frame, f'Green: {green}', green_points[0], font, font_scale, text_color,
                                      thickness, cv2.LINE_AA)
        image_with_text = cv2.putText(frame, f'Black: {black}', black_points[0], font, font_scale, text_color,
                                      thickness, cv2.LINE_AA)

        cv2.imshow('Cars Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tr = open('cars_info.txt', 'w', encoding='utf-8')
    string = f"""
    RED: {red};
    GREEN: {green};
    BLUE: {blue};
    BLACK: {black};
    """
    tr.write(string)


if __name__ == "__main__":
    main()
