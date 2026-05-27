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
                 p_oi_list: list, vertex_colors: list,
                 proc_scale: float = 0.5):
        """
        focal_length, cx, cy : camera intrinsics in ORIGINAL pixel coordinates.
        vertex_colors : list of n_f color range specs, one per vertex.
          Each spec is a list of one or more HSV ranges:
            [[h_min, s_min, v_min, h_max, s_max, v_max], ...]
          Multiple ranges are OR'd together (useful for red hue wraparound).
        proc_scale : downscale factor applied before feature detection (0 < s ≤ 1).
          0.5 → 320×240 processing (4× faster), accuracy unchanged for color blobs.
        """
        self.f  = float(focal_length)
        self.cx = float(cx)
        self.cy = float(cy)
        self._proc_scale = float(proc_scale)
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

        For each vertex, OR's its HSV ranges into a mask, finds the largest
        contour, and returns its centroid in normalized image coordinates.

        速度最適化:
          - proc_scale < 1.0 のとき検出前にダウンスケール（640×480 → 320×240 等）
            により色域マスク生成と輪郭検出が大幅に高速化される。
          - cv2.connectedComponentsWithStats + np.argwhere の代わりに
            cv2.findContours + cv2.moments を使用。
            C++ 内で完結するため Python GIL 保持時間が最小になる。

        img_bgr : (H, W, 3) uint8 BGR image (original resolution)
        Returns f ∈ R^{2*n_f} in normalized coordinates, or None if any
        vertex has no pixels matching its color ranges.
        """
        scale = self._proc_scale

        # ── ダウンスケール ─────────────────────────────────────────────
        if scale < 1.0:
            small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = img_bgr

        img_hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        features: list[float] = []

        for ranges in self._vertex_ranges:
            # ── マスク生成 ──────────────────────────────────────────────
            mask = np.zeros(small.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                cv2.bitwise_or(mask, cv2.inRange(img_hsv, lo, hi), dst=mask)

            # ── 最大輪郭のセントロイド ──────────────────────────────────
            # connectedComponentsWithStats (全画素ラベリング) の代わりに
            # findContours + moments を使用。C++ 完結で高速。
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] < 1.0:   # 面積がほぼゼロ = 有効な輪郭なし
                return None

            # スケール後座標をオリジナル画素座標に戻す
            u = (M['m10'] / M['m00']) / scale
            v = (M['m01'] / M['m00']) / scale

            features.append((u - self.cx) / self.f)
            features.append((v - self.cy) / self.f)

        return np.array(features)
