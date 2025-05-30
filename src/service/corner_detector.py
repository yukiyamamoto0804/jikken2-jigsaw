import shutil, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import cv2

from src.service.piece_division import PieceDivision

class ContourDetector:
    def __init__(self, image_path):
        """Initializes the ContourDetector with an image."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found or unable to load.")
        self.gray = None
        self.thresh = None
        self.contour = None
        self.image_with_contours = None

    def _preprocess(self):
        """Converts the image to grayscale and applies thresholding."""
        # Expand the input image
        self.image = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Convert the image to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        ret, self.thresh = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY)

    def _find_contour(self):
        """Finds contour in the thresholded image."""
        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        self.contour = [contours_sorted[0]]
        # print(f"Number of contours found: {len(self.contour)}")

    def _draw_contour(self):
        """Draws the found contour on the original image."""
        # Draw all contours on the original image
        self.image_with_contour = cv2.drawContours(self.image.copy(), self.contour, -1, (0, 255, 0), 2)

    def process(self):
        """Processes the image and displays the result."""
        self._preprocess()
        self._find_contour()
        self._draw_contour()

class CornerDetector:
    def __init__(
            self,
            input_dir="data/piece_division_tmp",
            output_dir="data/contours_trimmed",
            debug=False,
            data_init=True
            ):
        # input_dirがないときのエラーを書く####
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.debug = debug
        if data_init:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Error: input_dir does not exists {self.input_dir}")
        
        self.inputs = [] # 3の要素があるlist, 0番目にimgs, 1番目にcontours, 2番目にpathsを入れる
    
    def read_pieces(self, img_file = "imgs.json", contour_file = "contours.json", path_file = "paths.json"):
        img_path = self.input_dir / img_file
        contour_path = self.input_dir / contour_file
        path_path = self.input_dir / path_file
        for path in [img_path, contour_path, path_path]:
            if not path.exists():
                raise FileNotFoundError(f"指定されたパスが存在しません: {path}")
            if not path.is_file():
                raise ValueError(f"指定されたパスはファイルではありません: {path}")
            with open(path, "r", encoding="utf-8") as f:
                self.inputs.append(json.load(f))

    def detect_corners(self, window_size = 5, sigma = 3, thresh = 0.2):
        contours = self.input[1]
        paths = self.input[2]
        for contour, path, idx in zip(contours, paths, range(len(contours))):
            # 点列を1dフィッティングして傾きの角度をradで得る
            def tangent(xs):
                tangent, _ = np.polyfit(xs[0, :], xs[1, :], 1)
                #tangent, _ = np.polyfit(xs[:, 0], xs[:, 1], 1)
                rad = np.arctan(tangent)
                return np.pi-np.abs(rad) if (np.pi-np.abs(rad) < np.abs(rad)) else np.abs(rad)

            def apply_gaussian_filter(xs, sigma):
                return np.array([[x, y] for (x, y) in zip(gaussian_filter1d(xs[:, 0], sigma), gaussian_filter1d(xs[:,1], sigma))])

            piece = ContourDetector(path)
            piece.process()
            #print(f'piece{idx+1}: {piece.image.shape}')
            contour = np.array(piece.contour[0])

            window_size = (piece.image.shape[0] + piece.image.shape[1])//300
            sigma = (piece.image.shape[0] + piece.image.shape[1])//500
            xs = contour[:, 0, :].astype(np.float64)
            n_points = len(xs)
            # 入力点列にガウシアンフィルターをかける
            xs = apply_gaussian_filter(xs, sigma)
            # 前後window_size個の点列を取得
            xs_forward = np.lib.stride_tricks.sliding_window_view(xs, window_size, axis=0)
            xs_backward = np.lib.stride_tricks.sliding_window_view(np.roll(xs, window_size-1, axis=0), window_size, axis=0)

            # 前後それぞれの点列の傾きの角度を取得し、差分を取る
            tangent_forward = np.array(list(map(tangent, xs_forward)))
            tangent_forward = gaussian_filter1d(tangent_forward, sigma/2)
            tangent_backward = np.array(list(map(tangent, xs_backward)))
            tangent_backward = gaussian_filter1d(tangent_backward, sigma/2)

            # dtangent = np.array([np.pi-np.abs(t) if (np.pi-np.abs(t) < np.abs(t)) else t for t in (tangent_forward-tangent_backward)])
            dtangent = np.array([np.abs(t) for t in (tangent_forward-tangent_backward)])

            # dtangentにガウシアンフィルターをかける
            series = pd.Series(dtangent)
            dtangent = series.rolling(window=window_size, center=True).mean()

            corners = []
            dtangent_peaks, _ = find_peaks(dtangent)
            mask_window = len(dtangent)//200
            mask = np.zeros(len(dtangent_peaks)).astype(bool)
            for i in range(len(dtangent_peaks)):
                i_prev = i-1 if i != 0 else len(dtangent_peaks)-1
                i_next = i+1 if i != len(dtangent_peaks) -1 else 0
                if (dtangent[dtangent_peaks[i]] - dtangent[dtangent_peaks[i_prev]] > thresh\
                    and dtangent[dtangent_peaks[i]] - dtangent[dtangent_peaks[i_next]] > thresh):
                    mask[i] = True
            dtangent_peaks = dtangent_peaks[mask]
            if dtangent_peaks[0] > mask_window//2:
                dtangent_peaks = np.append(dtangent_peaks, window_size//2)
            dtangent_peaks = list(dtangent_peaks)
            dtangent_peaks.sort(key=lambda x: dtangent[x], reverse=True)
            corners = sorted(dtangent_peaks[:4])
            corners = np.array(corners)
            dtangent_corners = dtangent[corners]


            plt.figure(figsize=(100, 5))
            plt.plot(dtangent, label='dtangent')
            plt.xticks(np.arange(0, len(dtangent), 50))
            plt.scatter(corners, dtangent_corners)
            plt.legend()

            plt.savefig(f'tangent_{(idx+1):03d}.png')

            if len(corners) == 4:
                # 輪郭の表示
                contours_trimmed = [contour[corners[0]:corners[1]], \
                            contour[corners[1]:corners[2]], \
                            contour[corners[2]:corners[3]], \
                            np.concatenate((contour[corners[3]:], contour[:corners[0]]), axis=0)]

                image_with_contours = cv2.drawContours(piece.image, contours_trimmed, -1, (0, 255, 0), 2)

                cv2.imwrite(f'contour_{(idx+1):03d}.jpg', image_with_contours)
                np.savez(f'contour_{(idx+1):03d}.npz', arr0=contours_trimmed[0], arr1=contours_trimmed[1], arr2=contours_trimmed[2], arr3=contours_trimmed[3])
