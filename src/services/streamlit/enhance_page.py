import os
import sys

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

sys.path.append(os.path.dirname(os.path.abspath("src")))

import io
import logging

import requests

from src.services.bot.config_reader import settings


@st.cache_data(show_spinner=False, max_entries=10)
def upscale(img, scale_value):
    response = requests.request(
        "POST",
        f"{settings.BASE_SITE}/enhance/upscale",
        files={"image": img.getvalue()},
        params={"scale": scale_value},
    )
    return response.status_code, response.content


@st.cache_data(show_spinner=False, max_entries=10)
def deblur(img):
    response = requests.request(
        "POST", f"{settings.BASE_SITE}/enhance/deblur", files={"image": img.getvalue()}
    )
    return response.status_code, response.content


@st.cache_data(show_spinner=False, max_entries=10)
def denoise(img):
    response = requests.request(
        "POST", f"{settings.BASE_SITE}/enhance/denoise", files={"image": img.getvalue()}
    )
    return response.status_code, response.content


st.title("Обработка изображений")
c = st.container(border=True)
enhance_type = c.pills(
    "Выберите тип улучшения",
    options=["Повышение разрешения", "Удаление смазов", "Удаление шумов"],
    selection_mode="single",
)

if enhance_type == "Повышение разрешения":
    scale = c.segmented_control(
        label="Выберите степень повышения разрешения",
        options=["x2", "x4"],
        default="x2",
        selection_mode="single",
    )

input_img_bytes = c.file_uploader(
    "Загрузите изображение", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)

start_button = c.button("Обработать изображение")

if start_button and input_img_bytes is not None:
    with st.spinner("В процессе..."):
        if enhance_type == "Повышение разрешения":
            if scale == "x2":
                logging.info("Upscaling x2")
                status_code, output_img_bytes = upscale(input_img_bytes, 2)
            elif scale == "x4":
                logging.info("Upscaling x4")
                status_code, output_img_bytes = upscale(input_img_bytes, 4)
        elif enhance_type == "Удаление смазов":
            logging.info("Deblurring")
            status_code, output_img_bytes = deblur(input_img_bytes)
        elif enhance_type == "Удаление шумов":
            logging.info("Denoise")
            status_code, output_img_bytes = denoise(input_img_bytes)

        if status_code == 200:
            input_img = Image.open(input_img_bytes)
            output_img_bytes = io.BytesIO(output_img_bytes)
            output_img = Image.open(output_img_bytes)
            image_comparison(
                img1=input_img, img2=output_img, label1="До", label2="После"
            )
            st.download_button(
                "Скачать обработанное изображение",
                data=output_img_bytes,
                mime="image/png",
            )
        else:
            st.error("Упс! Что-то пошло не так")
            logging.error("Image processing error", exc_info=True)
