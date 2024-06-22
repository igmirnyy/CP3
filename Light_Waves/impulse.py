import numpy as np
from numba import njit, prange
import taichi

@njit
def rot(a: float) ->np.ndarray:
    """
    Generates 2D rotation matrix for angle a

    Args:
        a (float): angle

    Returns:
        np.ndarray: 2x2 rotation matrix
    """
    c = np.cos(a)
    s = np.sin(a)
    return np.array([[c, -s], [s, c]])


@njit
def wave_impulse(
    point: np.ndarray,  
    pos: np.ndarray,
    freq: float,  
    sigma: np.ndarray, 
                 
    ):
    """
    Calculates impulse from several waves at a point
    Args:
        point (np.ndarray): point where of the impulse is calculated
        pos (np.ndarray): center of the impulse
        freq (float): frequency of the impulse
        sigma (np.ndarray): size of Gauss bell via x and y axis
    :return: impulse at point
    """
    d = (point - pos) / sigma
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])



@njit(parallel=True)
def start_impulse(
        nx: int,
        ny: int,
        s_angle: np.ndarray,
        s_pos: np.ndarray,
        imp_freq: float,
        imp_sigma: np.ndarray
    ) -> np.ndarray:
    """Calculates starting impulse

    Args:
        nx (int): size of grid along x axis
        ny (int): size of grid along y axis
        s_rot (np.ndarray): angle of light source
        s_pos (np.ndarray): position of light source
        imp_freq (float): frequency of the impulse
        imp_sigma (np.ndarray): size of Gauss bell via x and y axis

    Returns:
        np.ndarray: _description_
    """
    s_rot = rot(s_angle)
    res = np.zeros((nx, ny, 3), dtype=np.float32)
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            uv = (np.array([i, j]) - 0.5 * np.array([nx, ny])) / ny
            uv = s_rot @ uv
            res[i, j, :] += wave_impulse(uv, s_pos, imp_freq, imp_sigma)
    return res


