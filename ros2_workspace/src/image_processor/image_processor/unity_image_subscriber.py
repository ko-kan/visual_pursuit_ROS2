import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from pathlib import Path

def _find_db_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "vertex_db" / "descriptors.npz"
        if candidate.exists():
            return candidate
        if (parent / "vertex_db").is_dir():
            return candidate  # ディレクトリはあるがファイル未生成
    return Path(__file__).resolve().parents[4] / "vertex_db" / "descriptors.npz"

DB_PATH = _find_db_path()
MATCH_DISTANCE_THRESHOLD = 50

# 頂点ID → BGR色（0:赤, 1:緑, 2:青, 3:黄）
VERTEX_COLORS = [
    (0,   0,   220),
    (0,   200,   0),
    (220,  80,  50),
    (0,   210, 210),
]

class UnityImageSubscriber(Node):
    def __init__(self):
        super().__init__('unity_image_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.listener_callback,
            10)
        self.subscription  # 防止用（unused variable warning）

        self.orb = cv2.ORB_create()
        self._load_descriptor_db()

    def _load_descriptor_db(self):
        if not DB_PATH.exists():
            self.get_logger().warn(f'descriptor db not found: {DB_PATH}')
            self.db_descriptors = None
            self.db_labels = None
            self.matcher = None
            return

        db = np.load(str(DB_PATH))
        db_all, db_labels = [], []
        for vi in range(4):
            key = f'vertex_{vi}'
            if key in db:
                descs = db[key]
                db_all.append(descs)
                db_labels.extend([vi] * len(descs))

        self.db_descriptors = np.vstack(db_all).astype(np.uint8)
        self.db_labels = np.array(db_labels)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.get_logger().info(
            f'DB loaded: {len(db_labels)} descriptors '
            f'({", ".join(f"v{i}={sum(1 for l in db_labels if l==i)}" for i in range(4))})'
        )

    def listener_callback(self, msg):
        image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.flip(image_bgr, 0)

        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(image_gray, None)

        image_with_kp = image_bgr.copy()

        if descriptors is not None and self.db_descriptors is not None:
            matches = self.matcher.match(descriptors, self.db_descriptors)

            counts = [0, 0, 0, 0]
            for kp, m in zip(keypoints, matches):
                if m.distance > MATCH_DISTANCE_THRESHOLD:
                    continue
                vertex_id = self.db_labels[m.trainIdx]
                color = VERTEX_COLORS[vertex_id]
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(image_with_kp, (x, y), 5, color, 1)
                cv2.circle(image_with_kp, (x, y), 2, color, -1)
                counts[vertex_id] += 1

            self.get_logger().info(
                f'matched: v0={counts[0]} v1={counts[1]} v2={counts[2]} v3={counts[3]}'
            )

        cv2.imshow("Unity Camera Feed", image_with_kp)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = UnityImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
