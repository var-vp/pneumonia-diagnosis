# pneumonia-prediction          https://pneumonia-diagnosis.streamlit.app/

### README for Pneumonia Detection from Chest X-rays

#### Project Overview:
This project utilizes machine learning to detect pneumonia from chest X-ray images. It includes two main components: a pneumonia classification model and an X-ray detector model. The application allows users to upload an image, classify if it's an X-ray, and predict the presence of pneumonia if applicable.

#### Features:
- **Upload Images**: Users can upload chest X-ray images (in JPG, JPEG, or PNG format) directly through the web interface.
- **Prediction**: The application uses a trained model to predict whether an uploaded image is an X-ray. If it is, it further predicts whether pneumonia is present.
- **User-Friendly Interface**: Built with Streamlit, the interface provides a seamless experience for users to interact with the prediction models.

#### Requirements:
Ensure you have the following dependencies installed:
- Python 3.8+
- `streamlit`
- `fastai`
- `PIL` (Pillow)

#### Setup Instructions:
1. Clone the repository:
   ```
   git clone https://github.com/khurshiduktamov/pneumonia-detection.git
   cd pneumonia-detection
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
4. Open your browser and navigate to `http://localhost:8501` to use the application.

#### Usage:
- Upload a chest X-ray image using the file uploader.
- Wait for the prediction results to appear.
- If the image is classified as an X-ray, the app will predict whether pneumonia is present.

#### Contributions:
Contributions to improve the models or enhance the application's features are welcome! Please fork the repository, make your changes, and submit a pull request.

---
