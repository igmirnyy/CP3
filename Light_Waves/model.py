from enum import StrEnum
from matplotlib import pyplot as plt
import numpy as np
from numba import njit, prange

from waves import start_impulse


class Polygon:
    def __init__(self, verteces: list[tuple[float, float]]) -> None:
        """
        Args:
            verteces (list[tuple[float, float]]): list of verteces of a polygon where 
                each vertex is connected to its neighbors 
        """
        self.verteces = np.array(verteces)
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Checks whether point is inside the polygon
        Args:
            point np.ndarray: array of size 2 with coordinates of the point 
        Returns:
            inside bool - True if point is in the polygon else False
        """
        return polygon_contains(self.verteces, point)
    

@njit
def polygon_contains(verteces: np.ndarray, point:np.ndarray) -> bool:
    """
    njit compilable backend for contains method
    """
    x, y = point
    inside = False
    p1_x = verteces[0][0]
    p1_y = verteces[0][1]
    for i in range(1, len(verteces) + 1):
        p2_x, p2_y = verteces[i % len(verteces)]
        # Check if the point is above the minimum y coordinate of the edge
        if y > min(p1_y, p2_y):
            # Check if the point is below the maximum y coordinate of the edge
            if y <= max(p1_y, p2_y):
                # Check if the point is to the left of the maximum x coordinate of the edge
                if x <= max(p1_x, p2_x):
                    # Calculate the x-intersection of the line connecting the point to the edge
                    x_intersection = (y - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x
    
                    # Check if the point is on the same line as the edge or to the left of the x-intersection
                    if p1_x == p2_x or x <= x_intersection:
                        # Flip the inside flag
                        inside = not inside
        p1_x, p1_y = p2_x, p2_y
    return inside
    

class Lense:
    def __init__(self, center: tuple[float, float], r: float, angle: float) -> None:
        """
        Generates a convex-concave lense where both sides are circles
        Args:
            r (float): radius of the convex side
            center (tuple[float, float]): center of the convex cide
            angle (float): angle of lense rotation in radians
        """
        self.convex_r = r
        self.concave_r = np.sqrt(2) * r
        self.convex_center = np.array(center)
        self.direction = np.array([np.cos(angle), np.sin(angle)])
        self.concave_center = self.convex_center - self.direction * self.concave_r/np.sqrt(2)
    
    def contains(self, point:np.ndarray) -> bool:
        """
        Checks whether point is inside the lense
        Args:
            point np.ndarray: array of size 2 with coordinates of the point 
        Returns:
            inside bool - True if point is in the lense else False
        """
        return lense_contains(self.concave_center, self.convex_center, self.concave_r, self.convex_r, point)

@njit
def lense_contains(concave_center, convex_center, concave_r, convex_r, point):
    """
    njit compilable backend for contains method
    """
    d1 = np.sum((point - convex_center) ** 2)
    d2 = np.sum((point - concave_center) ** 2)
    return d2 > (concave_r ** 2) and d1 < (convex_r**2)
    
class SimulationModeEnum(StrEnum):
    picture = "picture"
    video = "video"
    animation = "animation"


class Model:
    def __init__(
        self,
        width:int,
        height:int,
        c:float,
        dt:float,
        h:float,
        kappas: np.ndarray,
        freq: float,
        sigma: np.ndarray
    ) -> None:
        """
        A class for modeling behaviour of the light wave
        Args:
            width (int): width of the grid
            height (int): height of the grid
            c(float): speed of wave
            dt(float): time step
            acc(float| None): coefficient of accumulation
            h(float): grid step
            kappas (np.ndarray) refraction coefficients for red, green and blue
            freq (float): frequency of the impulse
            sigma (np.ndarray): size of Gauss bell via x and y axis
        """
        self.shape = width, height
        self.points = np.zeros(self.shape, dtype=np.float32)
        self.prism = Polygon(
            [
                [width//2, height//4],
                [width * 11//18 ,3 * height //4],
                [width * 7//9, 2 * height // 3],
                [2 * width//3, height // 6]
            ]
        )
        self.lense = Lense(
            [4*width//9, height//2],
            height//4,
            np.pi,
        )
        self.light_source = np.array([-1/2, -1/6])
        self.source_angle = 0
        self.c = c
        self.dt = dt
        self.h = h
        self.kappas = kappas
        self.freq= freq
        self.sigma= sigma
        self.setup()
    
    def setup(self) -> None:
        """
        Setups refraction coefficients of model and creating start impulse
        """
        Model._setup_pixels(
            self.points, self.prism.verteces,
            self.lense.concave_center, self.lense.convex_center,
            self.lense.concave_r, self.lense.convex_r,
        )
        self.kappa = (self.c * self.dt / self.h) * (self.points[None, ...] / self.kappas[:, None, None] + (1.0 - self.points[None, ...]))
        self.kappa = self.kappa.transpose((1, 2, 0))[:, ::-1]
        self.impulse = start_impulse(*self.shape, self.source_angle, self.light_source, self.freq, self.sigma)
        
    
    @staticmethod
    @njit(parallel=True)
    def _setup_pixels(points, prism_verteces, concave_center, convex_center, concave_r, convex_r):
        for i  in prange(len(points)):
            for j in range(len(points[i])):
                point = np.array([i, j])
                points[i, j] = int(lense_contains(concave_center, convex_center, concave_r, convex_r, point)
                                   or polygon_contains(prism_verteces, point)
                )


