from pathlib import Path
from warnings import warn
import os

from numba import jit as numba_jit
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix, linalg
from scipy import special
import numpy as np
import rustworkx as rx


# Pre-compute Gauss-Legendre nodes and weights (do this once globally)
_N_POINTS = 50
_x, _w = leggauss(_N_POINTS)
_THETA = np.pi/2 * (_x + 1)  # Transform from [-1,1] to [0,π]
_WEIGHTS = _w * np.pi/2


@numba_jit(nopython=True)
def _compute_integrals_jit(kappa_i, kappa_j, mu, R, beta, D, theta, weights):
    """
    Numerically integrate the angular-distance moments for one node pair.

    Uses Gauss-Legendre quadrature over theta in [0, pi] to evaluate the
    integrals of the (unnormalized) angular density of connected nodes on
    the D-sphere:

        rho(theta) ∝ sin(theta)^(D-1) / (1 + (R*theta / (mu*k_i*k_j)^(1/D))^beta)

    Parameters
    ----------
    kappa_i, kappa_j : float
        Hidden degrees (here, observed degrees) of the two nodes.
    mu : float
        Model parameter controlling the average degree.
    R : float
        Radius of the D-sphere.
    beta : float
        Inverse temperature of the connection probability.
    D : int
        Dimension of the sphere (embedding dimension minus one).
    theta : ndarray of shape (n_points,)
        Quadrature nodes mapped to [0, pi].
    weights : ndarray of shape (n_points,)
        Quadrature weights matching `theta`.

    Returns
    -------
    I_num : float
        Integral of theta * rho(theta) dtheta.
    I_den : float
        Integral of rho(theta) dtheta (normalization). The expected angular
        distance between connected nodes is I_num / I_den.
    """
    I_den = 0.0
    I_num = 0.0

    for i in range(len(theta)):
        t = theta[i]
        w = weights[i]

        # Compute integrand
        term = (R * t / (mu * kappa_i * kappa_j)**(1/D))**beta
        sin_term = np.sin(t)**(D-1)
        integrand_val = sin_term / (1 + term)

        I_den += w * integrand_val
        I_num += w * t * integrand_val

    return I_num, I_den



@numba_jit(nopython=True)
def expected_angular_distances(degrees, edges, mu, R, beta, D):
    N = degrees.shape[0]
    M = edges.shape[0]
    rows, cols, vals = np.zeros(2*M), np.zeros(2*M), np.zeros(2*M)
    for i, (v1, v2) in enumerate(edges):
        d1, d2 = degrees[v1], degrees[v2]
        if d1 > d2:
            d1, d2 = d2, d1

        I_num, I_den = _compute_integrals_jit(d1, d2, mu, R, beta, D, _THETA, _WEIGHTS)
        if I_den > 0.0:
            angle = I_num / I_den
        else:
            angle = np.pi / 2          # fallback if integrand underflowed
        weight = 2 * np.sin(angle / 2)

        rows[2*i] = v1;   cols[2*i] = v2;   vals[2*i] = weight
        rows[2*i+1] = v2; cols[2*i+1] = v1; vals[2*i+1] = weight

    # Kernel exponent, guarding against zero variance
    var = np.var(vals)
    if var < 1e-300:
        var = 1.0
    expo = np.square(vals) / var

    # Per-row minimum exponent (stabilization; cancels in the row normalization)
    row_min = np.full(N, np.inf)
    for k in range(2*M):
        r = int(rows[k])
        if expo[k] < row_min[r]:
            row_min[r] = expo[k]

    w = np.exp(-(expo - row_min[rows.astype(np.int64)]))

    strength = np.zeros(N)
    for k in range(2*M):
        strength[int(rows[k])] += w[k]

    out = np.empty(2*M)
    for k in range(2*M):
        out[k] = -w[k] / strength[int(rows[k])]

    return rows, cols, out


def build_weight_matrix(graph: rx.PyGraph, mu, R, beta, D):
    """
    Build the sparse matrix whose smallest eigenvectors give the
    Laplacian eigenmap embedding.

    Combines the off-diagonal entries from `expected_angular_distances`
    with a unit diagonal, giving a normalized (random-walk) Laplacian
    L = I - P weighted by expected angular distances.

    Parameters
    ----------
    graph : rustworkx.PyGraph
        Undirected input graph.
    mu, R, beta, D
        Model parameters; see `_compute_integrals_jit`.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (N, N)
        The weight (Laplacian) matrix.
    """
    rows, cols, vals = expected_angular_distances(
            np.array([graph.degree(u) for u in graph.node_indices()]),
            np.array(graph.edge_list()),
            mu,
            R,
            beta,
            D,
    )

    N = graph.num_nodes()
    rows = np.append(rows, np.arange(N))
    cols = np.append(cols, np.arange(N))
    vals = np.append(vals, np.ones(N))

    return csr_matrix((vals, (rows, cols)), shape=(N, N))


le_path = Path(__file__).parent.parent.parent.joinpath("laplacian-eigenmap")

def laplacian_eigenmap(edgelist_file, graph, dim, num_components, skip=False):
    """
    Compute a Laplacian eigenmap embedding, preferring the C++ implementation.

    Runs the external `laplacian-eigenmap` executable on the edge list and
    reads its output. If the executable fails or produces no output file, it
    falls back to the pure-Python `_laplacian_eigenmap`.

    Parameters
    ----------
    edgelist_file : str or Path
        Path to the graph's edge list file. The C++ output is read from
        `<edgelist_file stem>-le<dim>D.txt`.
    graph : rustworkx.PyGraph
        The same graph, used only for the Python fallback.
    dim : int
        Embedding dimension (must be >= 2).
    num_components : int
        Number of connected components, forwarded to the C++ program.
    skip : bool, default False
        If True and the output file already exists, reuse it instead of
        recomputing. If True but the file is missing, a warning is issued
        and the embedding is computed anyway.

    Returns
    -------
    dict
        - "pos": ndarray of shape (N, dim), the vertex coordinates.
        - "beta": float, estimated clustering parameter.
        - "r": float, estimated embedding radius.

    Raises
    ------
    ValueError
        If `dim < 2`.

    Warns
    -----
    UserWarning
        If `skip=True` but no output is available, or if the C++ program
        fails and the Python fallback is used.
    """
    if dim < 2:
        raise ValueError(f"Cannot embed in D<2 dimensions.")

    output_file = os.path.splitext(edgelist_file)[0] + f"-le{dim}D.txt"
    if skip and os.path.isfile(output_file):
        pass
    else:
        if skip:
            warn("Laplacian eigenmap output unavailable. Cannot skip its computation.")
        res = os.system(f"{le_path} {edgelist_file} {dim} {num_components}")
        if res != 0 or not os.path.isfile(output_file):
            warn("C++ Laplacian eigenmap failed. Using Python implementation.")
            return _laplacian_eigenmap(graph, dim)

    with open(output_file, "r") as file_stream:
        beta, embedding_radius = list(map(float, file_stream.readline()[2:].split()))

    return {
        "pos": np.loadtxt(output_file, comments="#", dtype=np.float64),
        "beta": beta,
        "r": embedding_radius,
    }


def _laplacian_eigenmap(graph, dim):
    """
    Pure-Python Laplacian eigenmap embedding into hyperbolic space.

    Estimates model parameters from the graph (beta = D + 1.5, with mu and
    R derived from N and the mean degree), builds the weighted Laplacian,
    and takes the eigenvectors with the smallest nonzero eigenvalues as
    angular coordinates. Radial coordinates are estimated from node
    degrees, and the result is mapped onto the hyperboloid model.

    Parameters
    ----------
    graph : rustworkx.PyGraph
        Undirected input graph.
    dim : int
        Embedding dimension; the sphere dimension is D = dim - 1.

    Returns
    -------
    dict
        - "pos": ndarray of shape (N, dim), where each row is the unit
          direction from the eigenvectors scaled by sinh(vertex radius).
        - "beta": float, the clustering parameter used.
        - "r": float, the estimated embedding radius (lower-bounded by 1).

    Raises
    ------
    ValueError
        If too few eigenvalues are found to extract `dim` nontrivial
        eigenvectors after discarding the (near-)zero ones.

    Notes
    -----
    The sparse eigensolver uses shift-invert (sigma=1e-12) to find the
    eigenvalues closest to zero. Eigenvalues with magnitude < 1e-15 are
    treated as zero modes (one per connected component) and skipped.
    Vertex radii are clipped to [0, 4/D * log(N)]; isolated vertices
    are placed at radius 0.
    """
    N = graph.num_nodes()
    M = graph.num_edges()

    # Embedding parameters
    D = dim-1
    beta = D + 1.5
    mu = (beta * special.gamma(D/2) * np.sin(D * np.pi / beta)) / (2 * np.pi**(1 + D/2) * 2 * M / N)
    R = (N * special.gamma((D + 1) / 2) / (2 * np.pi ** ((D + 1) / 2))) ** (1 / D)

    # Builds the weight matrix
    A = build_weight_matrix(graph, mu, R, beta, D)

    # Solves the eigenvalue problem.
    k = min(100 + D, N-1-1) # number of eigenvectors
    eigenvalues, eigenvectors = linalg.eigs(A, k=k, sigma=1e-12, which='LM')
    num_zeros = np.sum(np.abs(eigenvalues)<1e-15)

    if eigenvalues.shape[0] < num_zeros+dim:
        raise ValueError("Not enough eigenvalues were found.")

    degrees = np.array([graph.degree(u) for u in graph.node_indices()])
    kappa0 = np.min(degrees)
    if kappa0 == 0:
        kappa0 = 1
    embedding_radius = 2*np.log(2*R / (mu*kappa0**2)**(1/D))
    if embedding_radius < 1:
        embedding_radius = 1
    vertex_radii = embedding_radius - np.where(degrees==0, 0, 2/D*np.log(degrees/kappa0))
    vertex_radii = np.clip(vertex_radii, min=0., max=4/D*np.log(N))

    # Picks the dim first eigenvectors of smallest non-zero magnitude eigenvalue. The result is scaled
    # by the estimated hyperbolic radii to map onto the hyperboloid model.
    le_pos = np.real(eigenvectors[:, np.argsort(np.abs(eigenvalues))[num_zeros:num_zeros+dim]])
    le_pos = le_pos / np.expand_dims(np.sqrt(np.sum(np.square(le_pos), axis=1)), axis=-1)
    return {
        "pos": np.expand_dims(np.sinh(vertex_radii), axis=-1)*le_pos,
        "beta": beta,
        "r": embedding_radius,
    }
