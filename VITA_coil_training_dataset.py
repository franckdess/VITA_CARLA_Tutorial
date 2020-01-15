from carla import sensor
from carla.settings import CarlaSettings
"""
Configuration file used to collect the CARLA 100 data.
A more simple commented example can be found at coil_training_dataset_singlecamera.py
"""
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
POSITIONS = [ [29, 40], [13, 34], [87, 102], [27, 132], [44, 24],
              [26, 96], [67, 34], [1, 28], [134, 140], [9, 105],
              [129, 148], [16, 65], [16, 21], [97, 147], [51, 42],
              [41, 30], [107, 16], [47, 69], [95, 102], [145, 16],
              [64, 111], [47, 79], [69, 84], [31, 73], [81, 37],
              [57, 35], [116, 42], [47, 75], [143, 132], [8, 145],
              [107, 43], [111, 61], [105, 137], [72, 24], [77, 0],
              [80, 17], [32, 12], [64, 3], [32, 146], [40, 33],
              [127, 71], [116, 21], [49, 51], [110, 35], [85, 91],
              [114, 93], [30, 7], [110, 133], [60, 43], [11, 98], [96, 49], [90, 85],
              [27, 40], [37, 74], [97, 41], [110, 62], [19, 2], [138, 114], [131, 76],
              [116, 95], [50, 71], [15, 97], [74, 71], [50, 133],
              [23, 116], [38, 116], [101, 52], [5, 108], [23, 79], [13, 68]
            ]

FOV = 100

sensors_frequency = {'CentralRGB': 1, 'CentralSemanticSeg':  1, 'TopRGB': 1, 'TopSemanticSeg': 1}
sensors_yaw = {'CentralRGB': 0, 'CentralSemanticSeg': 0, 'TopRGB': 30.0, 'TopSemanticSeg': 30.0}

lat_noise_percent = 20
long_noise_percent = 20

NumberOfVehicles = [30, 60]  # The range for the random numbers that are going to be generated
NumberOfPedestrians = [50, 100]

set_of_weathers = [1, 3, 6, 8]

def make_carla_settings():
    """Make a CarlaSettings object with the settings we need."""

    settings = CarlaSettings()
    settings.set(
        SendNonPlayerAgentsInfo=True,
        SynchronousMode=True,
        NumberOfVehicles=30,
        NumberOfPedestrians=50,
        WeatherId=1)

    settings.set(DisableTwoWheeledVehicles=True)
    
    # --------------------------- CENTRAL CAMERA ---------------------------

    settings.randomize_seeds() # IMPORTANT TO RANDOMIZE THE SEEDS EVERY TIME
    camera0 = sensor.Camera('CentralSemanticSeg', PostProcessing='SemanticSegmentation')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)

    settings.add_sensor(camera0)
    camera0 = sensor.Camera('CentralRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)

    settings.add_sensor(camera0)

    # --------------------------- TOP CAMERA ---------------------------

    camera0 = sensor.Camera('TopSemanticSeg', PostProcessing='SemanticSegmentation')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(12.0, 0.0, 15.0)
    camera0.set_rotation(-90.0, 90.0, -90.0)

    settings.add_sensor(camera0)
    camera0 = sensor.Camera('TopRGB')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=FOV)
    camera0.set_position(12.0, 0.0, 15.0)
    camera0.set_rotation(-90.0, 90.0, -90.0)

    settings.add_sensor(camera0)


    return settings
