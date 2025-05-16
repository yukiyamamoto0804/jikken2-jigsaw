from pathlib import Path

import cv2
import numpy as np


def load_contours(npz_path: Path):
    data = np.load(str(npz_path), allow_pickle=True)
    contours = {}
    for key in data.files:
        try:
            arr = data[key]
            if arr.ndim == 3 and arr.shape[-1] == 2 and arr.size > 0:
                # reshape to OpenCV contour format (N,1,2) and int32
                contours[key.split("_")[-1]] = arr.reshape(-1, 1, 2).astype(np.int32)
        except Exception as e:
            print(f"⚠ Skipping {key} in {npz_path.name}: could not load ({e})")
    return contours


def compute_hu(contour):
    """Compute normalized Hu moments from an OpenCV contour."""
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m).flatten()
    # log scale as in OpenCV matchShapes convention
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


class ContoursMatch:
    def __init__(
        self,
        contour_dir="data/contours_trimmed",
        output_dir="data/contours_match_result",
        choose=1,
    ):
        self.contour_dir = Path(contour_dir)
        self.output_dir = Path(output_dir)
        self.choose = choose

    def contours_match(self):
        # 1) load all contours
        npz_files = sorted(self.contour_dir.glob("contour_*.npz"))
        if not npz_files:
            print("No .npz files found.")
            exit(1)

        # map piece index -> { edge_index -> contour }
        all_contours = {}
        for npz_file in npz_files:
            idx = int(npz_file.stem.split("_")[1])
            ctrs = load_contours(npz_file)
            if ctrs:
                all_contours[idx] = ctrs

        if self.choose not in all_contours:
            print(f"choose={self.choose} not found among contour files.")
            exit(1)

        # 2) compute Hu moments for every contour
        hu_table = {}
        for idx, edges in all_contours.items():
            hu_table[idx] = {}
            for edge_i, contour in edges.items():
                hu_table[idx][edge_i] = compute_hu(contour)

        # 3) for each edge of chosen piece, match to all other edges by L1
        chosen_edges = hu_table[self.choose]
        results = {edge_i: [] for edge_i in chosen_edges}

        for edge_i, hu_i in chosen_edges.items():
            for idx, edges in hu_table.items():
                if idx == self.choose:
                    continue
                for edge_j, hu_j in edges.items():
                    # L1 distance between Hu vectors
                    dist = np.sum(np.abs(hu_i - hu_j))
                    results[edge_i].append((idx, edge_j, dist))
            # sort and take top 5
            results[edge_i].sort(key=lambda x: x[2])
            results[edge_i] = results[edge_i][:5]

        # 4) output
        self.save_image(self, results)

    def save_image(self, results, piece_id):
        print(f"\n=== Top 5 L₁ matches for piece {self.choose} ===")
        for edge_i, matches in results.items():
            print(f"\n-- choose_{edge_i} --")
            for idx, edge_j, dist in matches:
                print(
                    f"piece {self.choose}_edge{edge_i} ↔ piece {idx}_edge{edge_j} : L1 = {dist:.6f}"
                )
                # このフォルダにpngファイルを保存
                output_path = (
                    self.output_dir / f"{piece_id}_{idx}_{edge_i}_{edge_j}.png"
                )


if __name__ == "__main__":
    contours_match = ContoursMatch()
    contours_match.contours_match()
