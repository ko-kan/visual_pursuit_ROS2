import cv2
import numpy as np
from pathlib import Path

DB_ROOT = Path(__file__).parent / "vertex_db"
NUM_VERTICES = 4

def main():
    orb = cv2.ORB_create()

    all_descriptors = {}

    for vi in range(NUM_VERTICES):
        vertex_dir = DB_ROOT / f"vertex_{vi}"
        image_paths = sorted(vertex_dir.glob("*.png"))

        if not image_paths:
            print(f"[WARNING] vertex_{vi}: 画像が見つかりません")
            continue

        descriptors_list = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARNING] 読み込み失敗: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is not None:
                descriptors_list.append(descriptors)

        if not descriptors_list:
            print(f"[WARNING] vertex_{vi}: 記述子が取得できませんでした")
            continue

        merged = np.vstack(descriptors_list)
        all_descriptors[f"vertex_{vi}"] = merged
        print(f"vertex_{vi}: {len(image_paths)} 枚 → 記述子 {merged.shape[0]} 個")

    if not all_descriptors:
        print("[ERROR] 保存する記述子がありません")
        return

    output_path = DB_ROOT / "descriptors.npz"
    np.savez(str(output_path), **all_descriptors)
    print(f"\n保存完了: {output_path}")

if __name__ == "__main__":
    main()
