# streamlit run src/services/streamlit/main.py
import logging

import streamlit as st
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

torch.classes.__path__ = []

start = st.Page("start_page.py", title="Стартовая страница")
enhance = st.Page("enhance_page.py", title="Обработка изображения")
pg = st.navigation([start, enhance])
st.set_page_config(page_title="Image enhancement")
pg.run()
