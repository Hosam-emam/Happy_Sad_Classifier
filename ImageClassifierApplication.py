import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import time

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main-title {
            color: #4CAF50;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .upload-box {
            border: 2px dashed #4CAF50;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #388E3C;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Function to Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'my_models/VibeClassifier.h5')

# --- Function to Classify Image ---
def classify_image(image_array):
    model = load_model()
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)[0][0]
    return prediction

# --- Function to Display Image with Prediction ---
def display_predicted_image(image_array, prediction):
    label = 'Happy 😊' if prediction <= 0.5 else 'Sad 😞'
    confidence = round((1 - prediction) * 100 if prediction <= 0.5 else prediction * 100, 2)
    
    st.image(image_array, caption=f'{label}: {confidence}%', use_container_width=True)

# --- App UI ---
st.markdown("<h1 class='main-title'>😊 Face Expression Classification 🎭</h1>", unsafe_allow_html=True)
st.write("Upload an image, and I'll predict whether it's **_Happy_** or **_Sad_**.")

upload = st.file_uploader(label="", type=['jpg', 'png', 'webp', 'jpeg'])

if upload is not None:
    img = image.load_img(upload, target_size=(256,256))
    img_array = np.array(img) / 255.0

    # --- Progress Bar Animation ---
    progress_bar = st.progress(0, text='🔍 Analyzing Emotions...')
    for percent in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(percent)

    # --- Perform Prediction ---
    prediction = classify_image(image_array=img_array)

    # --- Display Result ---
    display_predicted_image(image_array=img_array, prediction=prediction)

    st.success(f"**Detected Emotion:** {'Happy 😊' if prediction <= 0.5 else 'Sad 😢'}")
