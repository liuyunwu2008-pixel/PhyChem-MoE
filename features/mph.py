"""MPH: Multi-dimensional Persistence Homology → Persistence Images.

Uses GUDHI on Ubuntu server; scipy-based fallback for environments without GUDHI.

Fixes MPH encoder expression collapse via:
  - Multi-scale PI (concatenate multiple σ values)
  - Higher resolution (80×80 default, was 50×50)
  - Smaller σ default (0.5, was 1.0)
"""

from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial.distance import pdist, squareform


def _gudhi_persistence_diagram(
    coords: np.ndarray,
    max_dim: int = 1,
) -> np.ndarray:
    """Compute Vietoris-Rips persistence diagram via GUDHI.

    Returns combined (birth, death) pairs across H0 and H1.
    """
    import gudhi

    rips = gudhi.RipsComplex(points=coords)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()

    diagrams = []
    for dim in range(max_dim + 1):
        dgm = st.persistence_intervals_in_dimension(dim)
        if len(dgm) > 0:
            # Keep pairs with finite birth (death may be inf for essential classes)
            dgm = dgm[np.isfinite(dgm[:, 0])]
        else:
            dgm = np.zeros((0, 2))
        diagrams.append(dgm)

    combined = np.vstack([d for d in diagrams if len(d) > 0])
    if len(combined) == 0:
        return np.zeros((0, 2))
    return combined


def _diagram_to_persistence_image(
    diagram: np.ndarray,
    resolution: int,
    sigma: float,
) -> np.ndarray:
    """Convert (birth, death) pairs to persistence image via Gaussian weighting."""
    birth = diagram[:, 0]
    persistence = diagram[:, 1] - diagram[:, 0]
    persistence = np.clip(persistence, 1e-8, None)

    pi = np.zeros((resolution, resolution), dtype=np.float32)
    b_min, b_max = birth.min(), birth.max()
    p_min, p_max = persistence.min(), persistence.max()
    b_range = max(b_max - b_min, 1e-8)
    p_range = max(p_max - p_min, 1e-8)

    for b, p in zip(birth, persistence):
        bx = int((b - b_min) / b_range * (resolution - 1))
        py = int((p - p_min) / p_range * (resolution - 1))
        bx = np.clip(bx, 0, resolution - 1)
        py = np.clip(py, 0, resolution - 1)

        y_grid, x_grid = np.mgrid[0:resolution, 0:resolution]
        weights = np.exp(-((x_grid - bx) ** 2 + (y_grid - py) ** 2) / (2 * sigma**2))
        pi += weights

    return pi.astype(np.float32)


def _gudhi_persistence_image(
    coords: np.ndarray,
    resolution: int = 80,
    sigma_values: List[float] = None,
    max_dim: int = 1,
) -> np.ndarray:
    """Compute multi-scale Persistence Image via GUDHI.

    Computes PI at each sigma value and concatenates flattened results,
    providing multi-scale topological features that prevent encoder collapse.
    """
    if sigma_values is None:
        sigma_values = [0.25, 0.5, 1.0]

    diagram = _gudhi_persistence_diagram(coords, max_dim)
    if len(diagram) == 0:
        total_dim = resolution * resolution * len(sigma_values)
        return np.zeros(total_dim, dtype=np.float32)

    pis = []
    for sigma in sigma_values:
        pi = _diagram_to_persistence_image(diagram, resolution, sigma)
        pis.append(pi.flatten())

    return np.concatenate(pis).astype(np.float32)


def _scipy_persistence_image(
    coords: np.ndarray,
    resolution: int = 80,
    sigma_values: List[float] = None,
    n_filtrations: int = 30,
) -> np.ndarray:
    """Simplified multi-scale persistence image using scipy distance-based filtration."""
    if sigma_values is None:
        sigma_values = [0.25, 0.5, 1.0]

    dist = squareform(pdist(coords))
    n = len(coords)
    if n < 2:
        total_dim = resolution * resolution * len(sigma_values)
        return np.zeros(total_dim, dtype=np.float32)

    max_dist = dist.max() if dist.max() > 0 else 1.0
    radii = np.linspace(0, max_dist, n_filtrations)

    birth_death = []
    for i, r in enumerate(radii[:-1]):
        r_next = radii[i + 1]
        adj = (dist <= r).astype(np.float32)
        adj_next = (dist <= r_next).astype(np.float32)

        deg = adj.sum(axis=1)
        deg_next = adj_next.sum(axis=1)
        cc_new = int((deg_next > 0).sum() - (deg > 0).sum())

        tri = np.trace(np.linalg.matrix_power(adj, 3)) / 6
        tri_next = np.trace(np.linalg.matrix_power(adj_next, 3)) / 6

        birth_death.append([r, tri_next - tri + cc_new])

    points = np.array(birth_death)
    if len(points) == 0 or points.sum() == 0:
        total_dim = resolution * resolution * len(sigma_values)
        return np.zeros(total_dim, dtype=np.float32)

    pis = []
    for sigma in sigma_values:
        pi = _diagram_to_persistence_image(points, resolution, sigma)
        pis.append(pi.flatten())

    return np.concatenate(pis).astype(np.float32)


def compute_persistence_image(
    coords: np.ndarray,
    resolution: int = 80,
    sigma_values: Optional[List[float]] = None,
    use_gudhi: bool = True,
) -> np.ndarray:
    """Compute multi-scale Persistence Image from 3D molecular coordinates.

    Args:
        coords: (num_atoms, 3) array of atomic coordinates.
        resolution: PI grid resolution per scale (default 80 → 80×80 per scale).
        sigma_values: list of Gaussian kernel widths for multi-scale analysis.
            Default [0.25, 0.5, 1.0] — fine, medium, coarse scales.
        use_gudhi: Attempt GUDHI first, fall back to scipy.

    Returns:
        Flattened multi-scale Persistence Image vector of shape
        (resolution * resolution * len(sigma_values),).
    """
    if sigma_values is None:
        sigma_values = [0.25, 0.5, 1.0]

    if coords is None or coords.shape[0] < 2:
        total_dim = resolution * resolution * len(sigma_values)
        return np.zeros(total_dim, dtype=np.float32)

    if use_gudhi:
        try:
            return _gudhi_persistence_image(coords, resolution, sigma_values)
        except (ImportError, Exception):
            pass

    return _scipy_persistence_image(coords, resolution, sigma_values)
