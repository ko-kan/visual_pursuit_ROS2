import numpy as np
from .se3 import wedge


class Pose:
    """
    Holds an SE(3) pose g and integrates the dynamics

        ġ = -V̂_wc g + g V̂_second

    which covers both:
      - RRBM (eq 6.5): V_second = V^b_wo
      - Observer (eq 6.9): V_second = -u_e  (caller passes -u_e)
    """

    def __init__(self, g: np.ndarray):
        self.g = g.copy().astype(float)

    def step(self, dt: float, V_wc: np.ndarray, V_second: np.ndarray) -> None:
        """Euler integration step."""
        g_dot = -wedge(V_wc) @ self.g + self.g @ wedge(V_second)
        self.g = self.g + dt * g_dot
        self._project_so3()

    def _project_so3(self) -> None:
        """Projects rotation block back to SO(3) via SVD to prevent drift."""
        U, _, Vt = np.linalg.svd(self.g[:3, :3])
        D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
        self.g[:3, :3] = U @ D @ Vt
