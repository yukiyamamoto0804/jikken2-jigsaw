import streamlit as st
import os

page1 = st.Page("src/pages/upload_pieces.py", title="upload pieces", icon="📃")


def main() -> None:
    """アプリのエントリーポイント。ページルーターのように機能する。

    Returns:
        None
    """
    folder_setup()
    pg = st.navigation([page1])
    pg.run()

def folder_setup():
    # フォルダが存在しなければ作成
    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/puzzle_pieces/", exist_ok=True)
    os.makedirs("data/piece_transparent/", exist_ok=True)
    os.makedirs("data/result/", exist_ok=True)


if __name__ == "__main__":
    main()
