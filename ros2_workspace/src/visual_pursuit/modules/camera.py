import cv2
import numpy as np


class Camera:
    """
    Wraps two feature-extraction paths:
      - feature_from_model : camera model (eq 6.7), normalized coords
      - feature_from_image : HSV color blob detection → normalized coords

    All feature vectors have shape (2*n_f,): [x0, y0, x1, y1, ...].
    Normalized coordinates: x = (u - cx) / f, y = (v - cy) / f.

    Feature ordering matches p_oi_list: vertex_i ↔ vertex_colors[i].
    Returns None from feature_from_image if any vertex produces no pixels.
    """

    def __init__(self, focal_length: float, cx: float, cy: float,
                 p_oi_list: list, vertex_colors: list):
        """
        vertex_colors : list of n_f color range specs, one per vertex.
          Each spec is a list of one or more HSV ranges:
            [[h_min, s_min, v_min, h_max, s_max, v_max], ...]
          Multiple ranges are OR'd together (useful for red hue wraparound).
        """
        self.f = float(focal_length)
        self.cx = float(cx)
        self.cy = float(cy)
        self.p_oi_list = [np.array(p, dtype=float) for p in p_oi_list]
        self._vertex_ranges: list[list[tuple]] = []
        self.vertex_bgr_colors: list[tuple[int, int, int]] = []
        for spec in vertex_colors:
            if not isinstance(spec[0], list):
                spec = [spec]   # single range given directly
            self._vertex_ranges.append([
                (np.array(r[:3], dtype=np.uint8), np.array(r[3:], dtype=np.uint8))
                for r in spec
            ])
            # HSV 範囲の中心 H・最大 S/V から表示用 BGR カラーを生成
            r0 = spec[0]
            h_mid = int((r0[0] + r0[3]) / 2)
            hsv_px = np.uint8([[[h_mid, 255, 255]]])
            bgr = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0][0]
            self.vertex_bgr_colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))

    def feature_from_model(self, g_bar: np.ndarray) -> np.ndarray:
        """
        Predicts normalized feature coordinates from estimated pose ḡ.

        g_bar : estimated pose ḡ ∈ SE(3) as 4×4 (object → camera frame)
        Returns f_model ∈ R^{2*n_f} in normalized (X/Z, Y/Z) coordinates.
        """
        R = g_bar[:3, :3]
        t = g_bar[:3, 3]
        features = []
        for p_oi in self.p_oi_list:
            P = R @ p_oi + t
            features.extend([P[0] / P[2], P[1] / P[2]])
        return np.array(features)

    def feature_from_image(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """
        Detects feature centroids from a BGR image via HSV color thresholding.

        For each vertex, OR's its HSV ranges into a mask, then returns
        the centroid of that mask in normalized image coordinates.

        img_bgr : (H, W, 3) uint8 BGR image
        Returns f ∈ R^{2*n_f} in normalized coordinates, or None if any
        vertex has no pixels matching its color ranges.
        """
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        features = []

        for ranges in self._vertex_ranges:
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(img_hsv, lo, hi))

            # 最大連結成分のみを使用してノイズピクセルを除去
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            if n_labels < 2:   # 前景なし
                return None
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            pixels = np.argwhere(labels == largest)   # (N, 2): (row, col)

            v_mean = float(np.mean(pixels[:, 0]))   # row → pixel v
            u_mean = float(np.mean(pixels[:, 1]))   # col → pixel u
            features.append((u_mean - self.cx) / self.f)
            features.append((v_mean - self.cy) / self.f)

        return np.array(features)
