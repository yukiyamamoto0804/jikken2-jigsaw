from pathlib import Path

import cv2
import numpy as np


class PiecePositionDetector:
    def __init__(
        self,
        piece_dir="data/piece_transparent",
        complete_dir="data/complete_picture",
        output_dir="data/result",
        debug=False,
    ):
        self.piece_dir = Path(piece_dir)
        self.complete_dir = Path(complete_dir)
        self.output_dir = Path(output_dir)
        self.piece_dir.mkdir(parents=True, exist_ok=True)
        self.complete_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def main_process(self, piece_id, max_index):
        complete_img_path = self.complete_dir / f"{piece_id}.jpg"
        if complete_img_path.is_file():
            complete_img = cv2.imread(complete_img_path)
        else:
            raise FileNotFoundError(f"Input file not found: {complete_img_path}")
        for i in range(max_index):
            page_string = str(i + 1).zfill(3)
            piece_img_path = self.piece_dir / f"{piece_id}_{page_string}.png"
            if piece_img_path.is_file():
                piece_img = cv2.imread(piece_img_path, cv2.IMREAD_UNCHANGED)
                if self.debug:
                    cv2.imshow(
                        "Piece Image", piece_img
                    )  # ← 第1引数にウィンドウタイトル
                    cv2.waitKey(0)  # ← キー入力を待つ（表示されるのに必要）
                    cv2.destroyAllWindows()  # ← ウィンドウを閉じる処理
                self.jigsaw(complete_img, piece_img, piece_id, page_string)

    def jigsaw(self, imgA, imgB, piece_id, page_string):
        # --- アルファチャンネル処理 ---
        bgrB = imgB[:, :, :3]
        alphaB = imgB[:, :, 3]
        maskB = alphaB > 0

        # --- 特徴点検出（AKAZE） ---
        akaze = cv2.AKAZE_create()
        kpA, desA = akaze.detectAndCompute(imgA, None)
        kpB, desB = akaze.detectAndCompute(bgrB, None)

        # --- 特徴点マッチング（BF + Hamming） ---
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desB, desA)
        matches = sorted(matches, key=lambda x: x.distance)

        # --- 対応点抽出（上位N点） ---
        N_MATCHES = 30
        ptsB = np.float32([kpB[m.queryIdx].pt for m in matches[:N_MATCHES]]).reshape(
            -1, 1, 2
        )
        ptsA = np.float32([kpA[m.trainIdx].pt for m in matches[:N_MATCHES]]).reshape(
            -1, 1, 2
        )

        # --- 類似変換の推定（部分アフィン）---
        M, inliers = cv2.estimateAffinePartial2D(ptsB, ptsA, method=cv2.RANSAC)

        # --- ピース画像のワープ処理 ---
        hA, wA = imgA.shape[:2]
        warpedB = cv2.warpAffine(bgrB, M, (wA, hA), borderValue=(0, 0, 0))
        warpedMask = cv2.warpAffine(maskB.astype(np.uint8), M, (wA, hA))

        # --- 合成 ---
        result = imgA.copy()
        result[warpedMask > 0] = warpedB[warpedMask > 0]

        # --- 保存 & スコア出力 ---
        output_path = self.output_dir / f"{piece_id}_{page_string}.png"
        cv2.imwrite(output_path, result)
        score = np.mean([m.distance for m in matches[:N_MATCHES]])
        print(f"Similarity transform score (lower is better): {score:.2f}")


if __name__ == "__main__":
    piece_position_detector = PiecePositionDetector(debug=True)
    piece_position_detector.main_process(piece_id="piece", max_index=16)
