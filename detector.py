from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("src/application/services/yolo/weights/yolo11n.pt")

# Define path to video file
source = "C:/Users/yakov/Desktop/car_project/cars_det_V11/video/4.mp4"

# Run inference on the source
results = model(source, save=True,show=True)  # generator of Results objects