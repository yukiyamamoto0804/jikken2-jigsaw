import streamlit as st
from PIL import Image
import os

# ピースの写真をアップロードする
st.title("ジグソーパズルのピースをアップロード")

SAVE_DIR = "data/puzzle_pieces"

uploaded_image = st.file_uploader("ピースの画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    # 保存先のパス
    save_path = os.path.join(SAVE_DIR, uploaded_image.name)

    # 画像を保存
    image.save(save_path)

    st.success(f"画像を '{save_path}' に保存しました。")
    st.image(image, caption="アップロードしたピースの画像", use_column_width=True)