import numpy as np
from numba import njit, prange  # type: ignore


def init_boids(boids: np.ndarray, asp: float, vrange: tuple) -> np.ndarray:
    """Initialize random boids and their speed in array from uniform distribution"""
    n = boids.shape[0]
    rng = np.random.default_rng()
    low, high = vrange
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(low=low, high=high, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s
    return boids


@njit()
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """Calculate directions for arrows in boids model by propagating with speed and acceleration"""
    return np.hstack((boids[:, :2] - dt * boids[:, 2:4], boids[:, :2]))


@njit()
def norm(arr: np.ndarray):
    """Calculates norm via first axis"""
    return np.sqrt(np.sum(arr**2, axis=1))


@njit()
def center_point(array: np.ndarray):
    center = np.empty(2, dtype=np.float64)
    center[0] = np.mean(array[:, 0])
    center[1] = np.mean(array[:, 1])
    return center




@njit()
def vclip(v: np.ndarray, vrange: tuple[float, float]):
    norm = np.sqrt(v[:, 0] ** 2 + v[:, 1]**2)
    mask = norm > vrange[1]
    if np.any(mask):
        v[mask] *= (vrange[1] / norm[mask]).reshape(-1, 1)

@njit()
def propagate(boids: np.ndarray,
              dt: float,
              vrange: tuple[float, float],
              arange: tuple[float, float]):
    vclip(boids[:, 4:6], arange)
    boids[:, 2:4] += dt * boids[:, 4:6]
    vclip(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]



@njit()
def walls(boids: np.ndarray, asp: float, param: int):
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]
    order = param

    a_left = 1 / (np.abs(x) + c) ** order
    a_right = -1 / (np.abs(x - asp) + c) ** order

    a_bottom = 1 / (np.abs(y) + c) ** order
    a_top = -1 / (np.abs(y - 1.) + c) ** order

    return np.column_stack((a_left + a_right, a_bottom + a_top))

@njit()
def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
   x = np.clip((x - edge0) / (edge1 - edge0), 0., 1.)
   return x * x * (3.0 - 2.0 * x)

@njit()
def better_walls(boids: np.ndarray, asp: float, param: float):
    x = boids[:, 0]
    y = boids[:, 1]
    w = param

    a_left = smoothstep(asp * w, 0.0, x)
    a_right = -smoothstep(asp * (1.0 - w), asp, x)

    a_bottom = smoothstep(w, 0.0, y)
    a_top = -smoothstep(1.0 - w, 1.0, y)

    return np.column_stack((a_left + a_right, a_bottom + a_top))



@njit()
def distance(boids: np.ndarray) -> np.ndarray:
    """Calculates pairwise euclidean distance between boids"""
    p = boids[:, :2]
    n = p.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = p[i] - p[j]
            d = (v @ v)
            dist[i, j] = d
    dist = np.sqrt(dist)
    return dist


@njit()
def normalize(v):
    """Normalize vector to norm = 1"""
    v_norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / v_norm


@njit(parallel=True)
def visibility(boids: np.ndarray, perception: float, n_neighbours: int) -> np.ndarray:
    """
    Calculates pairwise euclidean distance between boids, angles and returns mask of visibility,
    implements a sector of angle width = (-arccos(angle), arccos(angle)) and radius = perception
    """
    N = boids.shape[0]
    D = np.empty(shape=(N, N), dtype=np.float64)
    mask = np.empty(shape=(N, N), dtype=np.bool8)
    for i in prange(N):
        D[i] = (boids[:, 0] - boids[i, 0]) ** 2 + (boids[:, 1] - boids[i, 1]) ** 2 
        farthest_neighbour = np.sort(D[i])[n_neighbours + 1]
        mask[i] = D[i] < min(farthest_neighbour, perception)
        mask[i,i] = False

    return mask


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """Implements cohesion component of acceleration via median center of group in sector"""
    center = center_point(boids[neigh_mask, :2])
    a = (center - boids[idx, :2]) / perception
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    """Implements separation component of acceleration via median within group in sector"""
    d = center_point(boids[neigh_mask, :2] - boids[idx, :2])
    return -d / ((d[0]**2 + d[1]**2) + 0.0001)


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """Implements median-based alingment component of acceleration within group in sector"""
    v_mean = center_point(boids[neigh_mask, 2:4])
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit()
def noise():
    """Implements of random noise in (-1, 1) interval for two coordinated, njit-compilable"""
    return np.random.uniform(-1,1,2)


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple) -> np.ndarray:
    """
    Implements boids visibility computation and acceleration computation via four different
    components - cohesion, alignment, separation, noise within sector of certain radius and angle
    """
    n = boids.shape[0]
    mask = visibility(boids, perception, 20)
    wal = better_walls(boids, asp, 1)
    for i in prange(n):
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

def simulation_step(boids: np.ndarray,
                    asp: float,
                    perception: float,
                    coefficients: np.ndarray,
                    vrange: tuple,
                    arange: tuple,
                    dt: float) -> None:
    """Implements full step of boids model simulation with updating their positions and propagation"""
    mask = flocking(boids, perception, coefficients, asp, vrange)
    propagate(boids, dt, vrange, arange)
    return mask