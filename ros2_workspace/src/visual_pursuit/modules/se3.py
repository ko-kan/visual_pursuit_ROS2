import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Maps R^3 vector to 3×3 skew-symmetric matrix."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def wedge(xi: np.ndarray) -> np.ndarray:
    """Maps R^6 body velocity [v; ω] to 4×4 se(3) matrix."""
    v, omega = xi[:3], xi[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = skew(omega)
    Xi[:3, 3] = v
    return Xi


def vee(Xi: np.ndarray) -> np.ndarray:
    """Maps 4×4 se(3) matrix to R^6 [v; ω]."""
    return np.array([Xi[0, 3], Xi[1, 3], Xi[2, 3],
                     Xi[2, 1], Xi[0, 2], Xi[1, 0]])


def sk(R: np.ndarray) -> np.ndarray:
    """Anti-symmetric part of a 3×3 matrix: (R - R^T) / 2."""
    return (R - R.T) / 2.0


def vee_skew(S: np.ndarray) -> np.ndarray:
    """Vee operator for a 3×3 skew-symmetric matrix → R^3."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def vec(g: np.ndarray) -> np.ndarray:
    """Maps SE(3) element g = (p, R) to R^6: [p; vee(sk(R))]."""
    R = g[:3, :3]
    p = g[:3, 3]
    return np.concatenate([p, vee_skew(sk(R))])


def Ad(g: np.ndarray) -> np.ndarray:
    """Adjoint representation Ad(g) of g ∈ SE(3), returns 6×6 matrix."""
    R = g[:3, :3]
    p = g[:3, 3]
    return np.block([
        [R,               skew(p) @ R],
        [np.zeros((3, 3)), R],
    ])


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map so(3) → SO(3) via Rodrigues formula."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) → SE(3)."""
    v, omega = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    R = exp_so3(omega)
    if theta < 1e-10:
        t = v
    else:
        K = skew(omega / theta)
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
        t = V @ v
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = t
    return g


def inv_se3(g: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) element: g^{-1} = (R^T, -R^T p)."""
    R = g[:3, :3]
    p = g[:3, 3]
    g_inv = np.eye(4)
    g_inv[:3, :3] = R.T
    g_inv[:3, 3] = -R.T @ p
    return g_inv


def phi(g: np.ndarray) -> float:
    """Storage function φ(g) = tr(I_3 - R) for g = (p, R) ∈ SE(3)."""
    return float(np.trace(np.eye(3) - g[:3, :3]))
