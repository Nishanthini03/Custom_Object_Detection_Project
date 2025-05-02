import streamlit as st
import torch
from PIL import Image
import pathlib
import sys
import numpy as np

# Patch PosixPath to WindowsPath for compatibility
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5', 'custom', path='weights/best.pt', source='local', force_reload=True)
    return model

model = load_model()

st.title("🐶 YOLOv5 Object Detection App (Cats & Dogs)")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Perform detection
    results = model(image)
    results.render()
    rendered_img = Image.fromarray(results.ims[0])

    # Create two columns to show images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📍 Detection Result")
        st.image(rendered_img, use_container_width=True)

st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #066839; /* Greenish background */
        color: #ffffff; /* Default text color */
    }

    /* Title */
    h1 {
        font-family: 'Arial Black', sans-serif;
        font-size: 2.8rem;
        color: #ffffff;
    }


    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #95D5B2;
        padding: 10px;
        border-radius: 10px;
    }


    /* Image styling */
    img {
        border: 4px solid #ffffff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    
    </style>
    """,
    unsafe_allow_html=True
)