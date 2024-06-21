from enum import StrEnum
import numpy as np

class SimulationModeEnum(StrEnum):
    picture = "picture"
    video = "video"
    animation = "animation"

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

h = 1.0 # grid size
c = 1.0 # speed of wave
dt = h / (c * 1.5) # time step
acc = 0.1 # accumulation coefficient (None or 0 for rendering light at the moment)
kappa = c * dt / h # base refraction coefficient
kappa_r = 1.30 
kappa_g = 1.35
kappa_b = 1.40
kappas = np.array([kappa_r, kappa_g, kappa_b]) #refraction coefficients for RGB
freq = 400.0 
sigma = np.array([0.005, 0.025])