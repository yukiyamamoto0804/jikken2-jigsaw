import os

import streamlit as st

from src.service.piece_division import PieceDivision
from src.service.piece_position_detector import PiecePositionDetector

page1 = st.Page(
    "src/pages/upload_complete_image.py", title="upload complete image", icon="📃"
)
page2 = st.Page("src/pages/upload_pieces.py", title="upload pieces", icon="📃")
page3 = st.Page("src/pages/reupload_pieces.py", title="reupload pieces", icon="📃")


def main() -> None:
    """アプリのエントリーポイント。ページルーターのように機能する。

    Returns:
        None
    """
    init_state()
    if st.session_state.page == "upload_complete_image":
        pg = st.navigation([page1])
        pg.run()

    if st.session_state.page == "upload_pieces":
        pg = st.navigation([page2])
        pg.run()

    if st.session_state.page == "reupload_pieces":
        pg = st.navigation([page3])
        pg.run()


def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "upload_complete_image"
    if "puzzle_saved" not in st.session_state:
        st.session_state.puzzle_saved = False
    if "puzzle_id" not in st.session_state:
        st.session_state.puzzle_id = None
    if "piece_division" not in st.session_state:
        st.session_state.piece_division = PieceDivision()
    if "piece_position_detector" not in st.session_state:
        st.session_state.piece_position_detector = PiecePositionDetector()


if __name__ == "__main__":
    main()
