import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = os.path.join("models", "best_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

st.title("🐱🐶 Cats vs Dogs Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"Prediction: 🐶 Dog ({prediction:.2f})")
    else:
        st.success(f"Prediction: 🐱 Cat ({1-prediction:.2f})")

    st.image(image)