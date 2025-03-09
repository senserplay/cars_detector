from src.application.enums.device import Device
from src.usecase.cars.count_two_region import CountTwoRegion

print(CountTwoRegion("/Users/alexandrbelyanin/Documents/PycharmProjects/v54_good/video/5.mp4", Device.cpu, True).execute())