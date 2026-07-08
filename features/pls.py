"""PLS: Persistent Laplacian Spectra extraction.

Computes a family of graph Laplacians across multiple filtration scales,
extracts eigenvalues, and returns the top-k spectrum.
"""

from typing import List
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform


def _build_laplacian(adj_matrix: np.ndarray) -> csr_matrix:
    """Build normalized graph Laplacian from adjacency matrix."""
    n = adj_matrix.shape[0]
    degrees = adj_matrix.sum(axis=1).flatten()
    # L = D - A (unnormalized)
    D = np.diag(degrees)
    L = D - adj_matrix
    # Normalized: L_sym = D^(-1/2) L D^(-1/2)
    d_inv_sqrt = np.zeros(n)
    mask = degrees > 1e-10
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_normalized = D_inv_sqrt @ L @ D_inv_sqrt
    return csr_matrix(L_normalized)


def compute_persistent_laplacians(
    coords: np.ndarray,
    top_k: int = 256,
    num_filtrations: int = 100,
) -> np.ndarray:
    """Compute Persistent Laplacian spectra.

    At each filtration radius (evenly spaced from 0 to max distance), build
    the thresholded adjacency matrix, compute the graph Laplacian, and extract
    eigenvalues. Concatenate across filtrations to form a spectral signature.

    Args:
        coords: (num_atoms, 3) atomic coordinates.
        top_k: number of top eigenvalues to retain per filtration.
        num_filtrations: number of filtration radii.

    Returns:
        (top_k,) array of aggregated eigenvalues or (num_filtrations * k,) if
        each filtration is kept separately. The default returns a global top-k
        across all filtrations.
    """
    n_atoms = coords.shape[0]
    if n_atoms < 3:
        return np.zeros(top_k, dtype=np.float32)

    dist = squareform(pdist(coords))
    max_dist = dist.max() if dist.max() > 0 else 1.0
    radii = np.linspace(max_dist / num_filtrations, max_dist, num_filtrations)

    all_evals: List[float] = []

    for r in radii:
        adj = (dist <= r).astype(np.float64)
        # Remove isolated nodes
        degrees = adj.sum(axis=1)
        if (degrees > 0).sum() < 2:
            continue

        try:
            L = _build_laplacian(adj)
            k = min(top_k, L.shape[0] - 2)
            if k <= 0:
                continue
            evals, _ = eigsh(L, k=k, which="SM", tol=1e-6)
            evals = np.sort(np.abs(evals))
            all_evals.extend(evals.tolist())
        except Exception:
            continue

    if not all_evals:
        return np.zeros(top_k, dtype=np.float32)

    all_evals_arr = np.array(all_evals, dtype=np.float32)
    # Take top-k across all filtrations
    if len(all_evals_arr) < top_k:
        padded = np.zeros(top_k, dtype=np.float32)
        padded[:len(all_evals_arr)] = np.sort(all_evals_arr)[::-1]
        return padded

    return np.sort(all_evals_arr)[-top_k:][::-1].astype(np.float32)
