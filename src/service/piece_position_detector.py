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

        complete_img = cv2.imread(str(complete_img_path))
        if complete_img is None:
            raise ValueError(f"画像の読み込みに失敗: {complete_img_path}")

        complete_img = self.apply_bilateral_filter(complete_img)
        complete_img = self.apply_clahe_to_v_channel(complete_img)

        for i in range(max_index):
            page_string = str(i + 1).zfill(3)
            piece_img_path = self.piece_dir / f"{piece_id}_{page_string}.png"
            if piece_img_path.is_file():
                piece_img = cv2.imread(piece_img_path, cv2.IMREAD_UNCHANGED)
                if self.debug:
                    cv2.imshow("Piece Image", piece_img)  # ← 第1引数にウィンドウタイトル
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
        imgA = self.resize_image(imgA, 1000)
        imgB = self.resize_image(imgB, 400)
        contours = self.get_contours(imgB)
        if self.debug:
            cv2.imshow("Piece Image", imgB)  # ← 第1引数にウィンドウタイトル
            cv2.waitKey(0)  # ← キー入力を待つ（表示されるのに必要）
            cv2.destroyAllWindows()  # ← ウィンドウを閉じる処理

        # --- アルファチャンネル処理 ---
        bgrB = imgB[:, :, :3]
        alphaB = imgB[:, :, 3]

        maskB = (alphaB > 0).astype(np.uint8)

        scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        best_score = float("inf")
        best_result = None

        result, score = self.try_match(imgA, bgrB, maskB, contours)
        if result is not None and score < best_score:
            best_score = score
            best_result = result

        for scale in scales:
            scaled_bgrB = cv2.resize(bgrB, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled_maskB = cv2.resize(maskB, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            result, score = self.try_match(imgA, scaled_bgrB, scaled_maskB, contours)
            if result is not None and score < best_score:
                best_score = score
                best_result = result

        if best_result is not None:
            output_path = self.output_dir / f"{piece_id}_{page_string}.png"
            cv2.imwrite(str(output_path), best_result)
            print(f"{piece_id}_{page_string}: Similarity score = {best_score:.2f}")
            if self.debug:
                cv2.imshow("Result", best_result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return output_path
        else:
            print(f"❌ マッチ失敗: {piece_id}_{page_string}")

    def try_match(self, imgA, imgB, maskB, contours):
        sift = cv2.SIFT_create()
        kpA, desA = sift.detectAndCompute(imgA, None)
        kpB_all, desB_all = sift.detectAndCompute(imgB, maskB)

        # print(imgA.shape)
        # kpA = []
        # kpB_all = []
        # desA = np.empty((0, 128))
        # desB_all = np.empty((0, 128))
        # for i in range(3):
        #     kpA_, desA_ = sift.detectAndCompute(imgA[:, :, i], None)
        #     kpB_all_, desB_all_ = sift.detectAndCompute(imgB[:, :, i], maskB)
        #     for item in kpA_:
        #         kpA.append(item)
        #     for item in kpB_all_:
        #         kpB_all.append(item)
        #     print(desA.shape, desA_.shape)
        #     print(desB_all.shape, desB_all_.shape)
        #     desA = np.concatenate((desA, desA_), axis=0)
        #     desB_all = np.concatenate((desB_all, desB_all_), axis=0)
        # kpA = tuple(kpA)
        # kpB_all = tuple(kpB_all)
        # print(len(kpB_all))
        # print(desB_all.shape)

        # 輪郭内の特徴点だけに限定
        kpB = self.filter_keypoints_inside_contours(kpB_all, contours)

        # 対応するdescriptorだけ取り出す
        idxs = [kpB_all.index(kp) for kp in kpB]
        desB = np.array([desB_all[i] for i in idxs])

        if desA is None or desB is None:
            print("None")
            return None, None

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(desB, desA, k=2)
        except:
            print("flann error")
            return None, None

        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 2:
            print("good_matches")
            return None, None

        # 対応点抽出（可能ならlen(matches)でクリップ）
        N_MATCHES = min(30, len(good_matches))

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        top_matches = good_matches[:N_MATCHES]
        print([top_matches[i].queryIdx for i in range(len(top_matches))])

        ptsB = np.float32([kpB[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
        ptsA = np.float32([kpA[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(ptsB, ptsA, method=cv2.RANSAC)

        if M is None or inliers is None:
            return None, None

        # ✅ 【チェック1】RANSAC inlier数
        # if np.sum(inliers) < 15:
        #     print("RANSAC inlier数")
        #     return None, None

        # ✅ 【チェック2】スケール制限
        # a, b, tx = M[0]
        # c, d, ty = M[1]
        # scale_x = np.sqrt(a**2 + c**2)
        # scale_y = np.sqrt(b**2 + d**2)
        # if not (0.85 <= scale_x <= 1.15 and 0.85 <= scale_y <= 1.15):
        #     print("scale")
        #     return None, None

        # ✅ 【チェック3】重心距離制限
        # hB, wB = imgB.shape[:2]
        # centerB = np.array([wB / 2, hB / 2, 1])
        # mapped_center = M @ centerB
        # hA, wA = imgA.shape[:2]
        # diag = np.sqrt(wA**2 + hA**2)
        # centerA = np.array([wA / 2, hA / 2])
        # distance = np.linalg.norm(mapped_center - centerA)
        # if distance > 0.1 * diag:
        #     print("distance")
        #     return None, None

        # 合格 → 合成
        hA, wA = imgA.shape[:2]
        warpedB = cv2.warpAffine(imgB, M, (wA, hA), borderValue=(0, 0, 0))
        warpedMask = cv2.warpAffine(maskB, M, (wA, hA))

        result = imgA.copy()
        result[warpedMask > 0] = warpedB[warpedMask > 0]

        for m in good_matches[:N_MATCHES]:
            ptA = tuple(np.round(kpA[m.trainIdx].pt).astype(int))  # 合成先（imgA上の点）
            ptB = tuple(
                np.round(cv2.transform(np.array([[kpB[m.queryIdx].pt]], dtype=np.float32), M)[0][0]).astype(int)
            )  # warp後の位置

            # 線を引く（青）、点を描く（赤）
            cv2.line(result, ptA, ptB, (255, 0, 0), 1)
            cv2.circle(result, ptA, 3, (0, 0, 255), -1)
            cv2.circle(result, ptB, 3, (0, 255, 0), -1)

        contours, _ = cv2.findContours(warpedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

        score = np.mean([m.distance for m in top_matches])
        return result, score

    def apply_bilateral_filter(self, img_bgr):
        return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    def apply_clahe_to_v_channel(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    piece_position_detector = PiecePositionDetector(debug=True, data_init=False)
    piece_position_detector.main_process_all(piece_id="9ce34836", max_index=16)
    # piece_position_detector.main_process_single(piece_id="b2f69267", page_string="001")
