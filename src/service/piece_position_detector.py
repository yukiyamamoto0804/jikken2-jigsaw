import shutil
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
        data_init=True,
    ):
        self.piece_dir = Path(piece_dir)
        self.complete_dir = Path(complete_dir)
        self.output_dir = Path(output_dir)
        if data_init:
            if self.piece_dir.exists():
                shutil.rmtree(self.piece_dir)
            if self.complete_dir.exists():
                shutil.rmtree(self.complete_dir)
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
        self.piece_dir.mkdir(parents=True, exist_ok=True)
        self.complete_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def main_process_all(self, piece_id, max_index):
        complete_img_path = self.complete_dir / f"{piece_id}.png"
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

    def main_process_single(self, piece_id, page_string):
        complete_img_path = self.complete_dir / f"{piece_id}.png"
        if complete_img_path.is_file():
            complete_img = cv2.imread(complete_img_path)
        else:
            raise FileNotFoundError(f"Input file not found: {complete_img_path}")
        piece_img_path = self.piece_dir / f"{piece_id}_{page_string}.png"
        if piece_img_path.is_file():
            piece_img = cv2.imread(piece_img_path, cv2.IMREAD_UNCHANGED)
            if self.debug:
                cv2.imshow("Piece Image", piece_img)  # ← 第1引数にウィンドウタイトル
                cv2.waitKey(0)  # ← キー入力を待つ（表示されるのに必要）
                cv2.destroyAllWindows()  # ← ウィンドウを閉じる処理
            output_path = self.jigsaw(complete_img, piece_img, piece_id, page_string)
        else:
            raise FileNotFoundError(f"Input file not found: {piece_img_path}")
        return output_path
    
    def resize_image(self, image, work_short_edge):
        h_full, w_full = image.shape[:2]
        scale = work_short_edge / min(h_full, w_full)
        w_work, h_work = int(w_full * scale), int(h_full * scale)
        img = cv2.resize(image, (w_work, h_work), interpolation=cv2.INTER_AREA)
        return img
    
    def get_contours(self, img):
        _, _, _, a = cv2.split(img)
        # 不透明な部分（α > 0）をマスクにする
        maskB = (a > 0).astype(np.uint8) * 255
        # 内側に20px縮小（モルフォロジー演算）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))  # 直径=2r+1でr=40
        inner_mask = cv2.erode(maskB, kernel)
        # 輪郭検出（外周輪郭のみ）
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return inner_contours
    
    # 特徴点のフィルタリング：輪郭の中にあるものだけを残す
    def filter_keypoints_inside_contours(self, keypoints, contours):
        filtered = []
        for kp in keypoints:
            pt = kp.pt
            # pointPolygonTest: 点が輪郭内なら >= 0
            if any(cv2.pointPolygonTest(cnt, pt, measureDist=False) >= 0 for cnt in contours):
                filtered.append(kp)
        return filtered

    def jigsaw(self, imgA, imgB, piece_id, page_string):
        imgA = self.resize_image(imgA, 2000)
        imgB = self.resize_image(imgB, 1000)
        contours = self.get_contours(imgB)
        if self.debug:
            cv2.imshow("Piece Image", imgB)  # ← 第1引数にウィンドウタイトル
            cv2.waitKey(0)  # ← キー入力を待つ（表示されるのに必要）
            cv2.destroyAllWindows()  # ← ウィンドウを閉じる処理

        # --- アルファチャンネル処理 ---
        bgrB = imgB[:, :, :3]
        alphaB = imgB[:, :, 3]
        maskB = alphaB > 0

        # --- 特徴点検出（AKAZE） ---
        akaze = cv2.AKAZE_create()
        kpA, desA = akaze.detectAndCompute(imgA, None)

        # AKAZE
        akaze = cv2.AKAZE_create()
        kpB_all, desB_all = akaze.detectAndCompute(bgrB, None)

        # 輪郭内の特徴点だけに限定
        kpB = self.filter_keypoints_inside_contours(kpB_all, contours)

        # 対応するdescriptorだけ取り出す
        idxs = [kpB_all.index(kp) for kp in kpB]
        desB = np.array([desB_all[i] for i in idxs])

        # --- 特徴点マッチング（BF + Hamming） ---
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desB, desA)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            return None
            # raise ValueError(f"マッチング点が少なすぎます: {len(matches)}点")

        # 対応点抽出（可能ならlen(matches)でクリップ）
        N_MATCHES = min(30, len(matches))

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

        # --- 対応点の可視化 ---
        for m in matches[:N_MATCHES]:
            ptA = tuple(np.round(kpA[m.trainIdx].pt).astype(int))  # 合成先（imgA上の点）
            ptB = tuple(np.round(cv2.transform(np.array([[kpB[m.queryIdx].pt]], dtype=np.float32), M)[0][0]).astype(int))  # warp後の位置

            # 線を引く（青）、点を描く（赤）
            cv2.line(result, ptA, ptB, (255, 0, 0), 1)
            cv2.circle(result, ptA, 3, (0, 0, 255), -1)
            cv2.circle(result, ptB, 3, (0, 255, 0), -1)

        # --- 保存 & スコア出力 ---
        output_path = self.output_dir / f"{piece_id}_{page_string}.png"
        cv2.imwrite(output_path, result)
        score = np.mean([m.distance for m in matches[:N_MATCHES]])
        print(f"Similarity transform score (lower is better): {score:.2f}")
        if self.debug:
            cv2.imshow("Piece Image", result)  # ← 第1引数にウィンドウタイトル
            cv2.waitKey(0)  # ← キー入力を待つ（表示されるのに必要）
            cv2.destroyAllWindows()  # ← ウィンドウを閉じる処理

        return output_path


if __name__ == "__main__":
    piece_position_detector = PiecePositionDetector(debug=True, data_init=False)
    piece_position_detector.main_process_all(piece_id="9ce34836", max_index=16)
    # piece_position_detector.main_process_single(piece_id="b2f69267", page_string="001")
