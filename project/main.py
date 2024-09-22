from video_processing import VideoProcessor
from region_selection import RegionSelector
from car_detection import CarDetector

def main():
    video_processor = VideoProcessor()
    region_selector = RegionSelector()
    car_detector = CarDetector()

    # Укажите путь до видео
    video_path = input('Укажите путь до видео: ')
    video_processor.load_video(video_path)

    # Выбор регионов
    regions = region_selector.select_regions(video_processor.get_frame())

    if not region_selector.validate_regions():
        print("Error: All regions must be defined.")
        return

    # Запуск процесса детекции машин
    car_detector.detect_cars(video_processor, regions)

if __name__ == "__main__":
    main()
