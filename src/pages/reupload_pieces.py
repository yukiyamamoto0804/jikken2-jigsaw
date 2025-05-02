from pathlib import Path

import streamlit as st
from PIL import Image

# ピースの写真をアップロードする
st.title("ピース確認ページ")

image_dir = Path("data/piece_transparent")
# for i in range(st.session_state.piece_division.idx):
#     # 画像を読み込む
#     image_path = image_dir / f"{st.session_state.puzzle_id}_{(i + 1):03d}.png"
#     if image_path.is_file():
#         image = Image.open(image_path)

#         # Streamlitで表示
#         st.image(image, caption="これは画像です", use_column_width=True)

# 画像の表示
for i in range(0, st.session_state.piece_division.idx, 4):
    cols = st.columns(4)  # 4つのカラム（列）を作成
    for j, col in enumerate(cols):
        if i + j < st.session_state.piece_division.idx:
            print(i + j, st.session_state.puzzle_id)
            img_path = image_path = (
                image_dir / f"{st.session_state.puzzle_id}_{(i + j + 1):03d}.png"
            )
            if image_path.is_file():
                image = Image.open(image_path)

                # Streamlitで表示
                col.image(image, width=150, caption="これは画像です")
