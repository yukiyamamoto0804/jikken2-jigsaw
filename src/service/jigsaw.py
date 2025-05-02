import cv2
import numpy as np


def jigsaw(imgA, imgB):
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
    ptsB = np.float32([kpB[m.queryIdx].pt for m in matches[:N_MATCHES]]).reshape(-1, 1, 2)
    ptsA = np.float32([kpA[m.trainIdx].pt for m in matches[:N_MATCHES]]).reshape(-1, 1, 2)

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
    cv2.imwrite('data/result/result_2.jpg', result)
    score = np.mean([m.distance for m in matches[:N_MATCHES]])
    print(f"Similarity transform score (lower is better): {score:.2f}")


if __name__ == "__main__":
    # --- 画像の読み込み ---
    imgA = cv2.imread('data/complete.jpg')  # 完成図
    imgB = cv2.imread('data/piece_transparent/piece_014.png', cv2.IMREAD_UNCHANGED)  # ピース（透過あり）
    jigsaw(imgA, imgB)
