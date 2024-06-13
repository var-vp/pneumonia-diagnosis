import streamlit as st
from PIL import Image
from model import load_model
from predict import predict
import pathlib

# Temporary redirect PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to get a prediction.")

model = load_model()

# Restore PosixPath
pathlib.PosixPath = temp

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Temporary redirect PosixPath to WindowsPath again during prediction
    pathlib.PosixPath = pathlib.WindowsPath

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_path = f"data/{uploaded_file.name}"
    image.save(img_path)

    # Restore PosixPath after prediction
    pathlib.PosixPath = temp
    
    label = predict(model, img_path)
    st.write(f"Prediction: {label}")
