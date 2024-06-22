import numpy as np
import taichi

from model import Model
import config as cfg



taichi.init(arch=taichi.cpu)

model = Model(cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT, cfg.c, cfg.dt, cfg.h, cfg.kappas, cfg.freq, cfg.sigma)

impulse_array_dict = {
            "past": model.impulse,
            "present": model.impulse,
            "future": model.impulse,
        }
        
light = taichi.Struct.field(
            {
            "past": taichi.math.vec3,
            "present": taichi.math.vec3,
            "future": taichi.math.vec3,
            }, 
            shape=model.shape
        )
kappa = taichi.Vector.field(3, dtype=taichi.f32, shape=model.shape)
        
light.from_numpy(impulse_array_dict)
kappa.from_numpy(model.kappa)
       

image = taichi.Vector.field(3, dtype=taichi.f32, shape=model.shape)


@taichi.func
def propagate() -> None:
    """
    One step of integrating the Euler equations for numerical refraction modelling

    Global args:
        light (Ltaichi.Struct.field): current light in model
        kappa (taichi.Vector): refraction coefficients
    """
    for x, y in light:
        light[x, y].future = kappa[x, y] ** 2 * (
                light[x - 1, y].present +
                light[x + 1, y].present +
                light[x, y - 1].present +
                light[x, y + 1].present -
                4 * light[x, y].present
        ) + 2 * light[x, y].present - light[x, y].past
    for x, y in light:
        light[x, y].past = light[x, y].present
        light[x, y].present = light[x, y].future


@taichi.func
def open_boundary() -> None:
    """
    Open boundary conditions for model of wave refraction

    Args:
        light (taichi.Struct): current light in model
        kappa (taichi.Vector): refraction coefficients
        SCREEN_WIDTH (int): number of points along x axis
        SCREEN_HEIGHT (int): number of points along y axis
    """
    for i, j in light:
        if i == 0:
            light[i, j].present = (light[i + 1, j].past
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i + 1, j].past - light[i, j].past)
                                  )
        elif i == cfg.SCREEN_WIDTH - 1:
            light[i, j].present = (light[i - 1, j].past
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i - 1, j].present - light[i, j].past)
                                  )
        if j == 0:
            light[i, j].present = (light[i, j + 1].past
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i, j + 1].present - light[i, j].past)
                                  )
        elif j == cfg.SCREEN_HEIGHT - 1:
            light[i, j].present = (light[i, j - 1].past
                                  + (kappa[i, j] - 1) / (kappa[i, j] + 1)
                                  * (light[i, j - 1].present - light[i, j].past)
                                  )



@taichi.func
def accumulate( ) -> None:
    """
    Accumulate the light movement in time

    Global args:
        image (taichi.Vector): accumulated light
        light (taichi.Struct): current light in model
        kappa (taichi.Vector): refraction coefficients
        SCREEN_WIDTH (int): number of points along x axis
        SCREEN_HEIGHT (int): number of points along y axis
        c (float): speed of wave
        dt (float): time step
        h (float): grid step
        acc (float): accumulation coefficient
    """
    for i, j in image:
        if 0 < i < cfg.SCREEN_WIDTH - 1 and 0 < j < cfg.SCREEN_HEIGHT - 1:
            image[i, j] += cfg.acc * taichi.abs(light[i, j].present) * kappa[i, j]
            

@taichi.kernel
def simulation_step_accumulate():
        """
        Updates all states of the model and accumulates light state
        Global args:
        image (taichi.Vector): accumulated light
        light (taichi.Struct): current light in model
        kappa (taichi.Vector): refraction coefficients
        SCREEN_WIDTH (int): number of points along x axis
        SCREEN_HEIGHT (int): number of points along y axis
        c (float): speed of wave
        dt (float): time step
        h (float): grid step
        acc (float): accumulation coefficient
        """
        open_boundary()
        propagate()

        accumulate()

@taichi.kernel
def simulation_step():
        """
        Updates all states of the model
        Global args:
        light (taichi.Struct): current light in model
        kappa (taichi.Vector): refraction coefficients
        SCREEN_WIDTH (int): number of points along x axis
        SCREEN_HEIGHT (int): number of points along y axis
        c (float): speed of wave
        dt (float): time step
        h (float): grid step
        acc (float): accumulation coefficient
        """
        open_boundary()
        propagate()