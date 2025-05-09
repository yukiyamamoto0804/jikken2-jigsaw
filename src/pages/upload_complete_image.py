import shutil
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image

# ピースの写真をアップロードする
st.title("ジグソーパズル全体の画像をアップロード")

uploaded_image = st.file_uploader(
    "パズル全体の画像をアップロード", type=["jpg", "png", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="パズル全体の画像をアップロード", use_container_width=True)

    if st.button("Go to Upload Pieces Page"):
        st.session_state.page = "upload_pieces"

if uploaded_image and st.session_state.puzzle_saved is False:
    st.session_state.puzzle_saved = True
    puzzle_id = uuid.uuid4().hex[:8]
    st.session_state.puzzle_id = puzzle_id
    st.session_state.piece_division.process_init()

    # 保存先のパス
    save_dir = Path("data/complete_picture")
    if save_dir.exists():
        shutil.rmtree(save_dir)  # ディレクトリごと削除
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{puzzle_id}.png"

    # 画像を保存
    image.save(save_path)

    st.success(f"画像を '{save_path}' に保存しました。")
