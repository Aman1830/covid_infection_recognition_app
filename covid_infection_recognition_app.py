import os
import time
import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# MUST BE FIRST Streamlit command
st.set_page_config(page_title="COVID Infection Recognition", layout="centered")

# Load the trained model
MODEL_PATH = 'classifier.keras'
MAPPING_PATH = 'ResultsMap.pkl'

@st.cache_resource
def load_classifier():
    return load_model(MODEL_PATH)

@st.cache_data
def load_result_map():
    with open(MAPPING_PATH, 'rb') as f:
        return pickle.load(f)

classifier = load_classifier()
ResultMap = load_result_map()

# Streamlit UI
st.title("🫁 COVID Infection Recognition")
st.write("Upload a CT scan image to predict whether it indicates COVID infection or not.")

uploaded_file = st.file_uploader("Choose a CT scan image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        prediction = classifier.predict(img_array, verbose=0)
        predicted_class = ResultMap[np.argmax(prediction)]

        # Show result
        st.image(img, caption="Uploaded CT Scan", use_column_width=True)
        st.subheader("Prediction:")
        st.write(f"🩺 **{predicted_class}**")
        st.write("Prediction probabilities:", prediction)

    except Exception as e:
        st.error(f"Error processing image: {e}")

