import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Ajanda - Image Cleaner", layout="centered")

st.title("🧹 Ajanda - Remove Unwanted Areas from Image")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("✏️ Draw on the image to mark unwanted area")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.4)",  # Red
        stroke_width=25,
        stroke_color="red",
        background_image=image,
        update_streamlit=True,
        height=img_np.shape[0],
        width=img_np.shape[1],
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        # Get the alpha channel from the canvas
        mask_rgba = canvas_result.image_data.astype(np.uint8)
        mask = mask_rgba[:, :, 3]  # Alpha channel

        # Convert alpha to binary mask
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # Inpaint using OpenCV
        inpainted_img = cv2.inpaint(img_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        st.subheader("🧼 Cleaned Image")
        st.image(inpainted_img, use_column_width=True)

        # Convert back to PIL and download
        result = Image.fromarray(inpainted_img)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("📥 Download Cleaned Image", data=byte_im, file_name="cleaned_image.png", mime="image/png")
