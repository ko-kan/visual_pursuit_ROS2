import numpy as np
from .se3 import skew


def image_jacobian(g_bar: np.ndarray, p_oi_list: list) -> np.ndarray:
    """
    Constructs image Jacobian J(ḡ) ∈ R^{2n_f × 6} (eq 6.15).

    Linearizes f(g_co) around g_co = ḡ: f_e ≈ J(ḡ) e_e.
    Works in normalized image coordinates (X/Z, Y/Z).

    g_bar  : estimated pose ḡ ∈ SE(3) as 4×4 matrix (object → camera frame)
    p_oi_list : list of n_f 3D feature positions in object frame, each R^3

    Each row-pair J_i = H_i @ R̄ @ [I_3 | -skew(p_oi)] (2×6)
    where H_i = [[1/Z, 0, -X/Z²], [0, 1/Z, -Y/Z²]] at estimated camera-frame point.
    """
    R_bar = g_bar[:3, :3]
    p_bar = g_bar[:3, 3]
    rows = []
    for p_oi in p_oi_list:
        P = R_bar @ p_oi + p_bar   # 3D point in estimated camera frame
        X, Y, Z = float(P[0]), float(P[1]), float(P[2])

        H_i = np.array([
            [1.0 / Z, 0.0,     -X / Z**2],
            [0.0,     1.0 / Z, -Y / Z**2],
        ])  # (2×3) projection Jacobian

        # (3×6) maps [p_ε; ω_ε] → δP in camera frame
        J_pose = np.hstack([R_bar, -R_bar @ skew(p_oi)])

        rows.append(H_i @ J_pose)  # (2×6)

    return np.vstack(rows)   # (2*n_f, 6)


def image_jacobian_pinv(J: np.ndarray) -> np.ndarray:
    """Pseudo-inverse J†(ḡ) ∈ R^{6 × 2n_f} (eq 6.16)."""
    return np.linalg.pinv(J)


def N_matrix(g_ce: np.ndarray) -> np.ndarray:
    """
    N matrix from eq 7.9: maps e = [e_c; e_e] → ν = [ν_c; ν_e].

      N = [[I_6,               0  ],
           [-Ad(e^{-ξ̂θ_ce}),  I_6]]

    Ad(e^{-ξ̂θ_ce}) = [[R_ce^T, 0], [0, R_ce^T]] (rotation-only adjoint).

    g_ce : control error g_ce = g_d^{-1} ḡ ∈ SE(3) as 4×4 matrix
    Returns N : (12, 12) matrix
    """
    R_ce_T = g_ce[:3, :3].T   # R_ce^{-1}
    Ad_rot = np.block([
        [R_ce_T,           np.zeros((3, 3))],
        [np.zeros((3, 3)), R_ce_T],
    ])  # Ad(e^{-ξ̂θ_ce}) : 6×6

    return np.block([
        [np.eye(6),  np.zeros((6, 6))],
        [-Ad_rot,    np.eye(6)],
    ])  # 12×12
