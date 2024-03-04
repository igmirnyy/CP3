SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
ASPECT_RATIO = SCREEN_WIDTH/SCREEN_HEIGHT
SCREEN_TITLE = "Boids"
N = 1000
dt = 0.05
PERCEPTION= 1 / 20
VELOCITY_RANGE = (0, 0.1)
ACCELERATION_RANGE = (0, 0.05)
COEFFITIENTS = {"Alignment": 0.4,
                "Cohesion": 0.6,
                "Separation": 0.4,
                "Walls": 17,
                "Noise": 0.01}
BOID_SCALE = 1
N_NEIGHBORS = 4
VIDEO = True
FRAMES = 3600