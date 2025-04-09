import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="Ajanda - Image Inpainting", layout="wide")

st.title("🖌️ Ajanda - Remove Unwanted Parts from Images")

with st.sidebar:
    st.header("Settings")
    stroke_width = st.slider("Brush Size", 10, 100, 40)
    stroke_color = st.color_picker("Brush Color", "#ffffff")
    bg_color = st.color_picker("Canvas Background", "#000000")
    drawing_mode = st.selectbox("Drawing Mode", ("freedraw", "rect", "circle", "transform"))
    real_time_update = st.checkbox("Update in Real Time", True)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("🎯 Draw over the area to remove")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        update_streamlit=real_time_update,
        height=image.height,
        width=image.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    if canvas_result.image_data is not None:
        mask = Image.fromarray((canvas_result.image_data[:, :, 3] > 0).astype(np.uint8) * 255)
        mask = mask.resize(image.size)

        if st.button("🪄 Remove Selected Area"):
            original_np = np.array(image)
            mask_np = np.array(mask)

            # Apply inpainting using OpenCV for now
            result = cv2.inpaint(original_np, mask_np, 3, cv2.INPAINT_TELEA)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                st.image(result, caption="Inpainted", use_column_width=True)
