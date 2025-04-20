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

# App title
st.title("🐾 Dog & Cat Object Detection!")

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #CAF0F8; /* Light Yellow Background */
        font-family: 'Roboto', sans-serif; /* Modern Font */
    }

    h1 {
        color: #4db6ac; /* Teal Accent Color */
        text-align: center;
        font-size: 3em;
        margin-bottom: 30px;
        text-shadow: 2px 2px #80cbc4; /* Subtle Text Shadow */
    }

    img {
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15); /* Enhanced Shadow */
        margin-bottom: 25px;
    }

    .stFileUploader {
        background-color: #f0f4c3; 
        border-radius: 10px;
        border: 1px solid #e6ee9c;
    }

    .stFileUploader label {
        color: #26a69a; /* Darker Teal */
        font-weight: bold;
        font-size: 1.1em;
    }

    .stMarkdown h2 {
        color: #00897b; /* Even Darker Teal */
        margin-top: 30px;
        border-bottom: 2px solid #4db6ac; /* Teal Underline */
        padding-bottom: 5px;
    }

    .streamlit-image-label {
        color: #64dd17; /* Lime Green for Labels */
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent background */
        padding: 5px 8px;
        border-radius: 5px;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a picture of your furry friend!", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Your Adorable Upload!", use_container_width=True)

    # Perform detection
    results = model(image)

    # Show result
    st.subheader("🐾 Dog / Cat Identifications:")
    results.render()
    rendered_img = Image.fromarray(results.ims[0])
    st.image(rendered_img, caption="✨ Here are the Detected Objects!", use_container_width=True)