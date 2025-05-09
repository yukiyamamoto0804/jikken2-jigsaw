import os
from pathlib import Path

import streamlit as st
from PIL import Image

# 縦横サイズを揃えるための高さ（例：150px幅に合わせた高さ、もしくは固定値で揃える）
MAX_HEIGHT = 150
DISPLAY_WIDTH = 150

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


def display_piece():
    # 画像の表示
    idx = 0
    while idx <= st.session_state.piece_division.idx:
        img_cols = st.columns(4)  # 4つのカラム
        # ボタン行
        btn_cols = st.columns(8)
        position_index = 0
        while position_index < 4:
            idx += 1
            if idx <= st.session_state.piece_division.idx:
                img_filename = f"{st.session_state.puzzle_id}_{(idx):03d}.png"
                img_path = image_dir / img_filename

                if img_path.is_file():
                    # PILで画像を読み込む
                    image = Image.open(img_path)

                    # 幅150に合わせてリサイズ（アスペクト比を維持しつつ高さ揃え）
                    aspect_ratio = image.height / image.width
                    new_height = int(DISPLAY_WIDTH * aspect_ratio)
                    if new_height > MAX_HEIGHT:
                        new_height = MAX_HEIGHT
                        # 高さをMAX_HEIGHTに合わせて幅を調整
                        new_width = int(MAX_HEIGHT / aspect_ratio)
                        image_resized = image.resize((new_width, MAX_HEIGHT))
                    else:
                        # 幅をDISPLAY_WIDTHに合わせて高さを調整
                        image_resized = image.resize((DISPLAY_WIDTH, new_height))

                    with img_cols[position_index]:
                        st.image(image_resized)

                    with btn_cols[position_index * 2]:
                        if st.button("削除", key=f"delete_{idx}"):
                            os.remove(img_path)
                            st.rerun()
                    with btn_cols[position_index * 2 + 1]:
                        if st.button("検証", key=f"check_{idx}"):
                            output_path = st.session_state.piece_position_detector.main_process_single(
                                piece_id=st.session_state.puzzle_id,
                                page_string=f"{(idx):03d}",
                            )
                            st.session_state.selected_image = output_path
                    position_index += 1
            else:
                break


display_piece()
# ポップアップとして画像を拡大表示
if st.session_state.selected_image:
    st.markdown("---")
    st.markdown("### 拡大表示（クリックで閉じる）")
    if st.button("✖️ 閉じる", key="close_popup"):
        st.session_state.selected_image = None
    else:
        st.image(Image.open(st.session_state.selected_image), use_container_width=True)
