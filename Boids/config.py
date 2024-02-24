SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
ASPECT_RATIO = SCREEN_WIDTH/SCREEN_HEIGHT
SCREEN_TITLE = "Boids"
N = 1000
dt = 0.1
PERCEPTION= 1 / 20
VELOCITY_RANGE = (0, 0.1)
ACCELERATION_RANGE = (0, 0.05)
COEFFITIENTS = {"Alignment": 0.3,
                "Cohesion": 0.3,
                "Separation": 0.4,
                "Walls": 7,
                "Noise": 0.01}
BOID_SCALE = 1
N_NEIGHBORS = 20
VIDEO = False
FRAMES = 1800