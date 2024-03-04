import numpy as np
from numba import njit, prange  # type: ignore


def init_boids(boids: np.ndarray, asp: float, vrange: tuple) -> np.ndarray:
    """Initialize random boids and their speed"""
    N = boids.shape[0]
    rng = np.random.default_rng()
    low, high = vrange
    boids[:, 0] = rng.uniform(0., asp, size=N)
    boids[:, 1] = rng.uniform(0., 1., size=N)
    alpha = rng.uniform(0, 2*np.pi, size=N)
    v = rng.uniform(low=low, high=high, size=N)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s
    return boids




@njit()
def center_point(array: np.ndarray):
    """Calculates center point of 2d np.array"""
    center = np.empty(2, dtype=np.float64)
    center[0] = np.mean(array[:, 0])
    center[1] = np.mean(array[:, 1])
    return center




@njit()
def vclip(v: np.ndarray, vrange: tuple[float, float]):
    """Clips array elements which norm is out of bounds"""
    norm = np.sqrt(v[:, 0] ** 2 + v[:, 1]**2)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)

@njit()
def propagate(boids: np.ndarray,
              dt: float,
              vrange: tuple[float, float],
              arange: tuple[float, float]):
    """Updates position and velocity of boids"""
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]



@njit()
def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
   x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
   return x * x * (3.0 - 2.0 * x)

@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    """Calculates walls impact on boids"""
    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))



@njit(parallel=True)
def visibility(boids: np.ndarray, perception: float, n_neighbours: int) -> np.ndarray:
    """
    Calculates 2d np.array where element i, j is True if boid j affects boid i otherwize false
    """
    N = boids.shape[0]
    perception = perception ** 2
    D = np.empty(shape=(N, N), dtype=np.float64)
    mask = np.empty(shape=(N, N), dtype=np.bool8)
    for i in prange(N):
        D[i] = (boids[:, 0] - boids[i, 0]) ** 2 + (boids[:, 1] - boids[i, 1]) ** 2 
        farthest_neighbour = np.sort(D[i])[n_neighbours]
        distance = min(farthest_neighbour, perception)
        mask[i] = D[i] <= distance
        mask[i,i] = False

    return mask


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """Caculates cohesion component of acceleration"""
    center = center_point(boids[neigh_mask, :2])
    a = (center - boids[idx, :2]) / perception
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """Calculates separation component of acceleration"""
    d = center_point(boids[neigh_mask, :2] - boids[idx, :2])
    return -d / ((d[0]**2 + d[1]**2) + 1e-8)


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """Calculates alignment component of acceleration"""
    v_mean = center_point(boids[neigh_mask, 2:4])
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit()
def noise():
    """Returns random noise uniformly disrtributed over (-1, 1)"""
    return np.random.uniform(-1,1,2)


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             N_neighbors: int,
             asp: float,
             vrange: tuple) -> np.ndarray:
    """
    Calculates boids visibility and updates acceleration using cohesion, alignment, separiotion, walls and noise
    """
    N = boids.shape[0]
    mask = visibility(boids, perception, N_neighbors)
    wal = better_walls(boids, asp, 1)
    for i in prange(N):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            ns = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i])
            ns = noise()
        boids[i, 4] = (coeffs[0] * coh[0]
                       + coeffs[1] * alg[0]
                       + coeffs[2] * sep[0]
                       + coeffs[3] * wal[i][0]
                       + coeffs[4] * ns[0])
        boids[i, 5] = (coeffs[0] * coh[1]
                       + coeffs[1] * alg[1]
                       + coeffs[2] * sep[1]
                       + coeffs[3] * wal[i][1]
                       + coeffs[4] * ns[1])
    return mask

def step(boids: np.ndarray,
                    asp: float,
                    perception: float,
                    coefficients: np.ndarray,
                    N_neighbors: int,
                    vrange: tuple,
                    arange: tuple,
                    dt: float) -> None:
    """Calculates boids acceleration and update their positions"""
    mask = flocking(boids, perception, coefficients, N_neighbors, asp, vrange)
    propagate(boids, dt, vrange, arange)
    return mask