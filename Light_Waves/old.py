import numpy as np
import taichi

from model import Model, SimulationModeEnum
import config as cfg



taichi.init(arch=taichi.cpu)
model = Model(cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT, cfg.c, cfg.dt, cfg.h, cfg.kappas, cfg.freq, cfg.sigma)

impulse_array_dict = {
            "past": model.impulse,
            "present": model.impulse,
            "future": model.impulse,
            "ahead":model.impulse,
        }
        
light = taichi.Struct.field(
            {
            "past": taichi.math.vec3,
            "present": taichi.math.vec3,
            "future": taichi.math.vec3,
            "ahead": taichi.math.vec3,
            }, 
            shape=model.shape
        )
kappa = taichi.Vector.field(3, dtype=taichi.f32, shape=model.shape)
        
light.from_numpy(impulse_array_dict)
kappa.from_numpy(model.kappa)
       

image = taichi.Vector.field(3, dtype=taichi.f32, shape=model.shape)


# @taichi.func
# def propagate() -> None:
#     """
#     One step of integrating the Euler equations for numerical refraction modelling

#     Args:
#         light (Ltaichi.Struct.field): current light in model
#         kappa (taichi.Vector): refraction coefficients
#         SCREEN_WIDTH (int): number of points along x axis
#         SCREEN_HEIGHT (int): number of points along y axis
#     """
#     for x, y in light:
#         light[x, y].future = kappa[x, y] ** 2 * (
#                 light[x - 1, y].present +
#                 light[x + 1, y].present +
#                 light[x, y - 1].present +
#                 light[x, y + 1].present -
#                 4 * light[x, y].present
#         ) + 2 * light[x, y].present - light[x, y].past
#     for x, y in light:
#         light[x, y].past = light[x, y].present
#         light[x, y].present = light[x, y].future


# @taichi.func
# def open_boundary() -> None:
#     """
#     Open boundary conditions for model of wave refraction

#     Args:
#         light (taichi.Struct): current light in model
#         kappa (taichi.Vector): refraction coefficients
#         nx (int): number of points along x axis
#         ny (int): number of points along y axis
#     """
#     for i, j in light:
#         if i == 0:
#             light[i, j].present = (light[i + 1, j].past
#                                   + (kappa[i, j] - 1) / (kappa[i, j] + 1)
#                                   * (light[i + 1, j].past - light[i, j].past)
#                                   )
#         elif i == cfg.SCREEN_WIDTH - 1:
#             light[i, j].present = (light[i - 1, j].past
#                                   + (kappa[i, j] - 1) / (kappa[i, j] + 1)
#                                   * (light[i - 1, j].present - light[i, j].past)
#                                   )
#         if j == 0:
#             light[i, j].present = (light[i, j + 1].past
#                                   + (kappa[i, j] - 1) / (kappa[i, j] + 1)
#                                   * (light[i, j + 1].present - light[i, j].past)
#                                   )
#         elif j == cfg.SCREEN_HEIGHT - 1:
#             light[i, j].present = (light[i, j - 1].past
#                                   + (kappa[i, j] - 1) / (kappa[i, j] + 1)
#                                   * (light[i, j - 1].present - light[i, j].past)
#                                   )



# @taichi.func
# def accumulate( ) -> None:
#     """
#     Accumulate the light movement in time

#     Args:
#         image (taichi.Vector): accumulated light
#         light (taichi.Struct): current light in model
#         kappa (taichi.Vector): refraction coefficients
#         nx (int): number of points along x axis
#         ny (int): number of points along y axis
#         c (float): speed of wave
#         dt (float): time step
#         h (float): grid step
#         acc (float): accumulation coefficient
#     """
#     for i, j in image:
#         if 0 < i < cfg.SCREEN_WIDTH - 1 and 0 < j < cfg.SCREEN_HEIGHT - 1:
#             image[i, j] += cfg.acc * taichi.abs(light[i, j].present) * kappa[i, j] / (cfg.c * cfg.dt / cfg.h)
            

# @taichi.kernel
# def simulation_step_accumulate():
#         """
#         Updates all states of the model
#             Args:
#         image (taichi.Vector): accumulated light
#         light (taichi.Struct): current light in model
#         kappa (taichi.Vector): refraction coefficients
#         nx (int): number of points along x axis
#         ny (int): number of points along y axis
#         c (float): speed of wave
#         dt (float): time step
#         h (float): grid step
#         acc (float): accumulation coefficient
#         """
#         open_boundary()
#         propagate()

#         accumulate()

# @taichi.kernel
# def simulation_step():
#         """
#         Updates all states of the model
#         Args:
#         image (taichi.Vector): accumulated light
#         light (taichi.Struct): current light in model
#         kappa (taichi.Vector): refraction coefficients
#         nx (int): number of points along x axis
#         ny (int): number of points along y axis
#         c (float): speed of wave
#         dt (float): time step
#         h (float): grid step
#         acc (float): accumulation coefficient
#         """
#         open_boundary()
#         propagate()
@taichi.func
def propagate():
    """One step of integrating the Euler equations for numerical refraction modelling"""
    for x, y in light:
        light[x, y].ahead = kappa[x, y] ** 2 * (
                light[x - 1, y].future +
                light[x + 1, y].future +
                light[x, y - 1].future +
                light[x, y + 1].future -
                4 * light[x, y].future
        ) + 2 * light[x, y].future - light[x, y].present


@taichi.func
def time_shift():
    """Shift arrays of points for one step forward in time"""
    for x, y in light:
        light[x, y].past = light[x, y].present
        light[x, y].present = light[x, y].future
        light[x, y].future = light[x, y].ahead


@taichi.func
def open_boundary():
    """Boundary conditions of open boundary for model of wave refraction"""
    for i, j in light:
        if i == 0:
            light[i, j].future = (light[i + 1, j].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i + 1, j].present - light[i, j].present)
                                  )
        elif i == cfg.SCREEN_WIDTH - 1:
            light[i, j].future = (light[i - 1, j].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i - 1, j].future - light[i, j].present)
                                  )
        if j == 0:
            light[i, j].future = (light[i, j + 1].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i, j + 1].future - light[i, j].present)
                                  )
        elif j == cfg.SCREEN_HEIGHT - 1:
            light[i, j].future = (light[i, j - 1].present
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i, j - 1].future - light[i, j].present)
                                  )


@taichi.func
def accumulate():
    """Accumulate the light movement in time and visualize it via Taichi GUI"""
    for i, j in image:
        if 0 < i < cfg.SCREEN_WIDTH - 1 and 0 < j < cfg.SCREEN_HEIGHT - 1:
            image[i, j] += cfg.acc * taichi.abs(light[i, j].future) * kappa[i, j] 

@taichi.kernel
def render():
    """Kernel for refraction model with Euler equations, implemented in Taichi"""
    open_boundary()
    propagate()
    time_shift()
    accumulate()



gui = taichi.GUI("Light Refraction", res=(cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT), fast_gui=True)

while gui.running:
    if gui.get_event(taichi.GUI.PRESS):
        if gui.event.key == taichi.GUI.ESCAPE:
            break
    render()
    gui.set_image(image)
    gui.show()
gui.close()
