from pathlib import Path
import cv2
import numpy as np

class PiecePositionDetector:
    def __init__(self, piece_dir="data/piece_transparent", complete_dir="data/complete_picture", output_dir="data/result", debug=False):
        self.piece_dir = Path(piece_dir)
        self.complete_dir = Path(complete_dir)
        self.output_dir = Path(output_dir)
        self.piece_dir.mkdir(parents=True, exist_ok=True)
        self.complete_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def main_process(self, piece_id, max_index):
        complete_img_path = self.complete_dir / f"{piece_id}.jpg"
        if not complete_img_path.is_file():
            raise FileNotFoundError(f"Input file not found: {complete_img_path}")

        complete_img = cv2.imread(str(complete_img_path))
        if complete_img is None:
            raise ValueError(f"画像の読み込みに失敗: {complete_img_path}")

        complete_img = self.apply_bilateral_filter(complete_img)
        complete_img = self.apply_clahe_to_v_channel(complete_img)

        for i in range(max_index):
            page_string = str(i + 1).zfill(3)
            piece_img_path = self.piece_dir / f"{piece_id}_{page_string}.png"
            if not piece_img_path.is_file():
                continue

            piece_img = cv2.imread(str(piece_img_path), cv2.IMREAD_UNCHANGED)
            if piece_img is None:
                continue

            self.jigsaw_multiscale(complete_img, piece_img, piece_id, page_string)

    def apply_bilateral_filter(self, img_bgr):
        return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    def apply_clahe_to_v_channel(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def jigsaw_multiscale(self, imgA, imgB, piece_id, page_string):
        bgrB = imgB[:, :, :3]
        alphaB = imgB[:, :, 3]
        maskB = (alphaB > 0).astype(np.uint8)

        scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        best_score = float('inf')
        best_result = None

        for scale in scales:
            scaled_bgrB = cv2.resize(bgrB, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled_maskB = cv2.resize(maskB, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            result, score = self.try_match(imgA, scaled_bgrB, scaled_maskB)
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
        else:
            print(f"❌ マッチ失敗: {piece_id}_{page_string}")

    def try_match(self, imgA, imgB, maskB):
        sift = cv2.SIFT_create()
        kpA, desA = sift.detectAndCompute(imgA, None)
        kpB, desB = sift.detectAndCompute(imgB, maskB)

        if desA is None or desB is None:
            return None, None

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(desB, desA, k=2)
        except:
            return None, None

        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return None, None

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        top_matches = good_matches[:30]

        ptsB = np.float32([kpB[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
        ptsA = np.float32([kpA[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(ptsB, ptsA, method=cv2.RANSAC)

        if M is None or inliers is None:
            return None, None

        # ✅ 【チェック1】RANSAC inlier数
        if np.sum(inliers) < 15:
            return None, None

        # ✅ 【チェック2】スケール制限
        a, b, tx = M[0]
        c, d, ty = M[1]
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)
        if not (0.85 <= scale_x <= 1.15 and 0.85 <= scale_y <= 1.15):
            return None, None

        # ✅ 【チェック3】重心距離制限
        hB, wB = imgB.shape[:2]
        centerB = np.array([wB / 2, hB / 2, 1])
        mapped_center = M @ centerB
        hA, wA = imgA.shape[:2]
        diag = np.sqrt(wA**2 + hA**2)
        centerA = np.array([wA / 2, hA / 2])
        distance = np.linalg.norm(mapped_center - centerA)
        if distance > 0.1 * diag:
            return None, None

        # 合格 → 合成
        hA, wA = imgA.shape[:2]
        warpedB = cv2.warpAffine(imgB, M, (wA, hA), borderValue=(0, 0, 0))
        warpedMask = cv2.warpAffine(maskB, M, (wA, hA))

        result = imgA.copy()
        result[warpedMask > 0] = warpedB[warpedMask > 0]

        contours, _ = cv2.findContours(warpedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

        score = np.mean([m.distance for m in top_matches])
        return result, score

if __name__ == "__main__":
    detector = PiecePositionDetector(debug=True)
    detector.main_process(piece_id="piece", max_index=40)