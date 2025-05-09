import shutil
from pathlib import Path

import streamlit as st
from PIL import Image

# ピースの写真をアップロードする
st.title("ジグソーパズルのピースをアップロード")

uploaded_image = st.file_uploader(
    "ピースの画像をアップロード", type=["jpg", "png", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image)

    # 保存先のパス
    save_dir = Path("data/puzzle_pieces")
    if save_dir.exists():
        shutil.rmtree(save_dir)  # ディレクトリごと削除
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{st.session_state.puzzle_id}.png"

    # 画像を保存
    image.save(save_path)

    st.success(f"画像を '{save_path}' に保存しました。")
    st.image(image, caption="アップロードしたピースの画像", use_container_width=True)

    if st.button("ピース確認ページへ"):
        st.session_state.piece_division.extract_multi_pieces(st.session_state.puzzle_id)
        st.session_state.page = "reupload_pieces"
