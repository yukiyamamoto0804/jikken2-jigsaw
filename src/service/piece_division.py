import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class PieceDivision:
    def __init__(
        self,
        input_dir="data/puzzle_pieces",
        output_dir="data/piece_transparent",
        debug=False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_pieces_masked(self, piece_id, work_short_edge=800):
        image_path = self.input_dir / f"{piece_id}.png"
        # 1. 画像読み込み & リサイズ（作業用画像）
        img_full = cv2.imread(image_path)
        h_full, w_full = img_full.shape[:2]
        scale = work_short_edge / min(h_full, w_full)
        w_work, h_work = int(w_full * scale), int(h_full * scale)
        img = cv2.resize(img_full, (w_work, h_work), interpolation=cv2.INTER_AREA)

        # 2. 前処理（グレースケール → ぼかし → エッジ検出 → 膨張）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # 3. 輪郭抽出（外側のみ）
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        idx = 0

        for cnt in contours:
            # 4. 小さすぎるものは無視
            x_w, y_w, w_box_w, h_box_w = cv2.boundingRect(cnt)
            if w_box_w < 30 or h_box_w < 30:
                continue

            # 5. 元画像スケールに変換（ピクセル座標）
            cnt_orig = (cnt / scale).astype(np.int32)
            x, y, w_box, h_box = cv2.boundingRect(cnt_orig)

            # 6. ROI（元画像から切り出し）
            roi = img_full[y : y + h_box, x : x + w_box]

            # 7. マスク作成（正確な輪郭をROIローカル座標に変換）
            cnt_local = cnt_orig.copy()
            cnt_local[:, 0, 0] -= x
            cnt_local[:, 0, 1] -= y

            mask = np.zeros((h_box, w_box), dtype=np.uint8)
            cv2.drawContours(mask, [cnt_local], -1, color=255, thickness=-1)

            # 8. 最小トリミング（マスク範囲だけに限定）
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            roi_cropped = roi[y_min : y_max + 1, x_min : x_max + 1]
            mask_cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

            # 9. RGBA画像として保存（アルファチャンネル追加）
            roi_rgb = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2RGB)
            alpha = mask_cropped  # 255: ピース, 0: 透明
            rgba = np.dstack([roi_rgb, alpha])

            idx += 1
            out_path = self.output_dir / f"piece_{idx:03d}.png"
            Image.fromarray(rgba).save(out_path)

        print(f"{idx} 個の透明背景付きピースを切り出して保存しました。")


if __name__ == "__main__":
    piece_division = PieceDivision(debug=False)
    piece_division.extract_pieces_masked("piece")
