import numpy as np

PROFILE_ON = False

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

FRAME_DECIMATED_WIDTH = 320#428#640
FRAME_DECIMATED_HEIGHT = 180#240#360

#FRAME_DECIMATED_WIDTH = 1280
#FRAME_DECIMATED_HEIGHT = 720

#FULL_FRAME_WIDTH = 1920
#FULL_FRAME_HEIGHT = 1080
FULL_FRAME_WIDTH = FRAME_WIDTH
FULL_FRAME_HEIGHT = FRAME_HEIGHT

MAX_FLOAT32 = 2.0**31 - 1

DISTANCE_THRESHOLD = 50.

MAX_DISTANCE = 8000.
MIN_DISTANCE = 500.

MAX_DISTANCE_METERS = (MAX_DISTANCE/1000.)
MIN_DISTANCE_METERS = (MIN_DISTANCE/1000.)

MIN_CONTOUR_AREA = 500
MIN_HOLE_AREA = 50

DECIMATION_FACTOR = 4

KP_SIZE = 10
BACKGROUND = [255, 255, 255]
COLOR_MAP = np.array([
    [255, 106, 0],
    [255, 0, 0],
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [127, 201, 255],
    [127, 255, 255],
    [165, 255, 127],
    [127, 255, 197],
    [214, 127, 255],
    [161, 127, 255],
    [107, 63, 127],
    [63, 73, 127],
    [63, 127, 127],
    [109, 127, 63],
    [255, 127, 237],
    [127, 63, 118],
    [0, 74, 127],
    [255, 0, 110],
    [0, 127, 70],
    [127, 0, 0],
    [33, 0, 127],
    [127, 0, 55],
    [38, 127, 0],
    [127, 51, 0],
    [64, 64, 64],
    [73, 73, 73],
    [0, 0, 0],
    [191, 168, 247],
    [192, 192, 192],
    [127, 63, 63],
    [127, 116, 63],
    BACKGROUND
]);

COMPRESS_FRONT = [
#head
    (41, 38),
    (42, 38),
    (40, 38),
    (39, 38),
#neck
    (10, 9),
    (8, 9),
    (16, 9),
    (15, 9),
    (17, 9),
#shoulders (assume front)
    (14, 12),
    (13, 11),
#body (assume front)
    (6, 0),
    (4, 1),
    (7, 2),
    (5, 3),
#feet (assume front)
    (25, 26),
    (24, 27),
];

BIMODAL_SET = set([
    28,33,32,31,30,
    34,35,36,37,29,
    18,19,20,21,22,23,
    26,27, 11, 12
])

BACKGROUND_INDEX = COLOR_MAP.shape[0] - 1
INNER_WINDOW = 190
MARGIN = 30
WINDOW_SIZE = INNER_WINDOW + 2*MARGIN
NUM_BLOCKS = 10
SIZE_BLOCK = (INNER_WINDOW / NUM_BLOCKS)

#ignore body part if there are less than MIN_COUNT pixels
MIN_COUNT = 180#150
