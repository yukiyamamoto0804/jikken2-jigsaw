import streamlit as st

page1 = st.Page("src/pages/upload_pieces.py", title="upload pieces", icon="📃")


def main() -> None:
    """アプリのエントリーポイント。ページルーターのように機能する。

    Returns:
        None
    """
    pg = st.navigation([page1])
    pg.run()


if __name__ == "__main__":
    main()
