import streamlit as st
from PIL import Image
from model import load_model, load_xray_detector
from predict import predict, is_xray
import pathlib

# Hide warnings
st.set_option('client.showErrorDetails', False)

# Set dark theme
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
    }
    .stButton>button {
        background-color: #444;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the page title
st.set_page_config(page_title="X-Ray Pneumonia Detection")

# Temporary redirect PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Function to load models
@st.cache_resource()
def load_models():
    model = load_model()  # Load pneumonia prediction model
    xray_detector = load_xray_detector()  # Load X-ray detector model
    return model, xray_detector

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to get a prediction.")

# Load models
model, xray_detector = load_models()

# Function to predict and display results
def predict_and_display_results(image, uploaded_file):
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("\nChecking if the image is an X-ray...")
    
    # Save uploaded image temporarily
    img_path = f"data/{uploaded_file.name}"
    image.save(img_path)
    
    if is_xray(xray_detector, img_path):
        st.write("✅ Image is an X-ray. Classifying for pneumonia...")
        label = predict(model, img_path)
        st.write(f"**🩺 Prediction: {label}**")
    else:
        st.write("❌ Uploaded image is not an X-ray. Please upload a chest X-ray image.")

# Handle file upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        predict_and_display_results(image, uploaded_file)
    except:
        st.write("⚠️ Invalid image file. Please try again.")

# Restore PosixPath after prediction
pathlib.PosixPath = temp
