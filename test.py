from src.application.enums.device import Device
from src.application.schemas.region_coordinates import Coordinates, RegionCoordinates
from src.usecase.cars.count_two_region import CountTwoRegion


region_coordinates = [RegionCoordinates(**region) for region in [{'region_id': 1, 'region_name': 'start',
                                                                      'vertices': [Coordinates(x=45, y=696),
                                                                                   Coordinates(x=359, y=516),
                                                                                   Coordinates(x=867, y=822),
                                                                                   Coordinates(x=392, y=1079)]},
                                                                     {'region_id': 2, 'region_name': 'end',
                                                                      'vertices': [Coordinates(x=1164, y=401),
                                                                                   Coordinates(x=1469, y=235),
                                                                                   Coordinates(x=1838, y=420),
                                                                                   Coordinates(x=1488, y=657)]}]]
#Теперь можно просто название видоса, но он должен лежать в video
#print(CountTwoRegion("video_2025-03-31_17-32-37.mp4", Device.cuda, False).execute())
print(CountTwoRegion("video_2025-03-31_17-32-38.mp4", Device.cuda, False).execute())