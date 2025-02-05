import streamlit as st
from PIL import Image
from model import load_model, load_xray_detector
from predict import predict, is_xray
import pathlib

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection", 
    page_icon="🩺", 
    layout="centered"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .uploadedImage {
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Temporary redirect PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Function to load models
@st.cache_resource()
def load_models():
    model = load_model()
    xray_detector = load_xray_detector()
    return model, xray_detector

# Title & Description
st.markdown("""
    # 🩺 Pneumonia Detection from Chest X-rays
    ### Upload a chest X-ray image to get a prediction.
""")

# Load models
model, xray_detector = load_models()

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Upload a chest X-ray image")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='🖼️ Uploaded Image', use_column_width=True, output_format="auto", clamp=True, channels="RGB", class_="uploadedImage")
        
        st.write("🔍 Checking if the image is an X-ray...")
        img_path = f"data/{uploaded_file.name}"
        image.save(img_path)

        if is_xray(xray_detector, img_path):
            st.write("✅ Image is an X-ray. Classifying for pneumonia...")
            label = predict(model, img_path)
            
            if label.lower() == "pneumonia":
                st.error(f"🔴 **Prediction: {label}**")
            else:
                st.success(f"🟢 **Prediction: {label}**")
        else:
            st.warning("⚠️ Uploaded image is not an X-ray. Please upload a valid chest X-ray image.")
    except Exception as e:
        st.error("❌ Invalid image file. Please try again.")

# Restore PosixPath after prediction
pathlib.PosixPath = temp
