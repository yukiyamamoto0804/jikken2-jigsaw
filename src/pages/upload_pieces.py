import streamlit as st
from PIL import Image

# ピースの写真をアップロードする
st.title("ジグソーパズルのピースをアップロード")

uploaded_image = st.file_uploader("ピースの画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="アップロードしたピースの画像", use_column_width=True)