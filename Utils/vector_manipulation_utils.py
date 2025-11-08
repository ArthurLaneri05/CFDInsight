import numpy as np
from typing import Union, Tuple


def gram_schmidt_ortogonalization(x_axis: Union[list, np.ndarray], 
                                  z_axis: Union[list, np.ndarray]
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates an ortonormal reference frame from an x_axis and z_axis, 
    not strictly perpendicular to each other.
    """

    # Ortogonalize the x axis using the Gram–Schmidt process
    z_axis = np.array(z_axis, dtype=float)
    z_axis /= np.linalg.norm(z_axis)

    # Ortogonalize the x axis using the Gram–Schmidt process
    x_axis = np.array(x_axis, dtype=float)
    x_axis -= np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    # Create y_axis
    y_axis = np.cross(z_axis, x_axis)

    return x_axis, y_axis, z_axis


def euler_angles_from_axes_deg(x, y, z) -> np.ndarray:
    """
    Given vectors of a frame,
    return the XYZ intrinsic Euler angles (deg)
    that rotate this frame into the canonical basis.
    Assumes x, y, z are already orthonormal (right-handed).

    Conventions:
    - Intrinsic XYZ (about body axes) == Extrinsic ZYX (about fixed axes).
    - Angles returned as [alpha (X), beta (Y), gamma (Z)] in degrees.
    """

    # source basis (columns are basis vectors)
    S = np.column_stack((x, y, z))

    # rotation that maps S -> I
    R = S.T

    # Extract angles for intrinsic XYZ (== extrinsic ZYX)
    # beta from -R[2,0] = sin(beta)
    s = np.clip(-R[2,0], -1.0, 1.0)
    eps = 1e-12

    if abs(s) < 1.0 - eps:
        beta  = np.arcsin(s)
        alpha = np.arctan2(R[2,1], R[2,2])  # about X
        gamma = np.arctan2(R[1,0], R[0,0])  # about Z
    else:
        # Gimbal lock: |beta| ≈ 90°
        beta = np.pi/2 * np.sign(s)

        # Disambiguation choice to match expected output:
        # set alpha = 0 and solve gamma from the first row.
        alpha = 0.0
        if s > 0:   # beta = +90°
            # With alpha=0: R[0,1] = -sin(gamma), R[0,2] =  cos(gamma)
            gamma = np.arctan2(-R[0,1], R[0,2])
        else:       # beta = -90°
            # With alpha=0: R[0,1] =  sin(gamma),  R[0,2] = -cos(gamma)
            gamma = np.arctan2(R[0,1], -R[0,2])

    return np.degrees([alpha, beta, gamma])


def get_WindToBody_RotMat(alpha_deg: float, 
                          beta_deg: float) -> np.ndarray:
    """ Outputs the rotation matrix from the wind frame to the body frame.
        (x_b forward, y_b right, z_b down)."""
    a = np.deg2rad(alpha_deg); b = np.deg2rad(beta_deg)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    return np.array([
        [-ca*cb, -ca*sb, -sa],
        [-sb,    cb,     0  ],
        [sa*cb,  -sa*sb, -ca]
    ])


def order_points_CCW(pts: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Return the points sorted along a path (approximately CCW for closed loops).
    Heuristic: build a k-NN graph (k=2 by default), walk from an endpoint; 
    fallback: 1D PCA sort if the walk doesn't visit all points.

    Parameters
    ----------
    pts : (N, D) array_like
        Input points (D=2 or 3).
    k : int, optional
        Number of nearest neighbors to wire in the graph (>=1).

    Returns
    -------
    pts_sorted : (N, D) ndarray
        Points reordered along the inferred path.
    """
    pts = np.asarray(pts)
    N = pts.shape[0]
    if N <= 2:
        return pts.copy()

    # pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt(np.einsum('ijk,ijk->ij', d, d))
    np.fill_diagonal(D, np.inf)

    # k-NN undirected adjacency
    k = max(1, min(k, max(1, N - 1)))
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]
    nbrs = [set() for _ in range(N)]
    for i in range(N):
        for j in nn[i]:
            j = int(j)
            nbrs[i].add(j)
            nbrs[j].add(i)

    # endpoints (deg=1) -> open path; else closed loop
    deg = np.array([len(n) for n in nbrs])
    endpoints = np.flatnonzero(deg == 1)

    # start: endpoint if open, else lexicographic min
    if endpoints.size:
        e = endpoints
        start = int(e[np.lexsort((pts[e, 2] if pts.shape[1] > 2 else np.zeros_like(e),
                                  pts[e, 1], pts[e, 0]))[0]])
    else:
        start = int(np.lexsort((pts[:, 2] if pts.shape[1] > 2 else np.zeros(N),
                                pts[:, 1], pts[:, 0]))[0])

    # walk the graph
    order = np.empty(N, dtype=int)
    visited = np.zeros(N, dtype=bool)
    cur, prev = start, -1
    for t in range(N):
        order[t] = cur
        visited[cur] = True
        candidates = [j for j in nbrs[cur] if not visited[j]]
        if prev in candidates and len(candidates) > 1:
            candidates.remove(prev)
        if not candidates:
            if t == N - 1:
                break
            # jump to the closest unvisited point to keep the path going
            rem = np.where(~visited)[0]
            cur = int(rem[np.argmin(D[order[t], rem])])
        else:
            cd = np.array([D[cur, j] for j in candidates])
            nxt = int(candidates[int(np.argmin(cd))])
            prev, cur = cur, nxt

    # fallback if not all visited: project on first PC and sort
    if not visited.all():
        c = pts.mean(axis=0, keepdims=True)
        U, _, _ = np.linalg.svd((pts - c), full_matrices=False)
        proj = (pts - c) @ U[:, [0]]  # (N,1)
        order = np.argsort(proj.ravel(), kind='mergesort')

    return pts[order]


def cluster_by_radius(X: np.ndarray, eps: float, min_pts: int = 1) -> list[np.ndarray]:
    """
    Euclidean (ε) clustering with a simple connected-components DFS.
    Returns one np.ndarray per cluster and DISCARDs small components (noise).

    Parameters
    ----------
    X : (N, D) array_like
        Points in 2D/3D (any D works).
    eps : float
        Connection radius. Points are linked if their distance <= eps.
    min_pts : int, optional
        Minimum number of points required to keep a component as a cluster.
        Components with size < min_pts are treated as noise and dropped.

    Returns
    -------
    clusters : list[np.ndarray]
        Each element is an array view of the points belonging to one cluster,
        in the ORIGINAL input order.
    """
    X = np.asarray(X, dtype=float)
    N = len(X)
    if N == 0:
        return []

    # Pairwise squared distances (O(N^2) in memory/time)
    d = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', d, d)

    # ε-neighborhood adjacency (exclude self)
    neigh = (d2 <= eps * eps)
    np.fill_diagonal(neigh, False)

    visited = np.zeros(N, dtype=bool)
    clusters: list[np.ndarray] = []

    # Flood-fill / DFS over the ε-neighborhood graph
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        members = []

        while stack:
            j = stack.pop()
            members.append(j)

            # Unvisited neighbors of j
            nbrs = np.flatnonzero(neigh[j] & (~visited))
            if nbrs.size:
                visited[nbrs] = True
                # Extend DFS frontier
                stack.extend(nbrs.tolist())

        # Keep only components with at least min_pts elements
        if len(members) >= min_pts:
            # Preserve original input order inside each cluster
            members = np.sort(np.asarray(members, dtype=int))
            clusters.append(order_points_CCW(X[members]))

    return clusters