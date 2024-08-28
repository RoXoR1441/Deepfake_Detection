import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the model
model_path = 'meso4_model.h5'  # Adjust this path to your actual model path
model = load_model(model_path)

# Set the dimensions for the input image
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

# Define a function to make predictions
def predict(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return prediction

# Streamlit app
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="centered",
    initial_sidebar_state="auto",
)

# Page title
st.title("🕵️‍♂️ Deepfake Detection")
st.write("### Upload an image to check if it's real or a deepfake.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(image_dimensions['height'], image_dimensions['width']))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict and display result
    st.write("")
    with st.spinner('Classifying...'):
        prediction = predict(image)
        if prediction >= 0.5:
            st.success(f"Prediction: Real")
        else:
            st.error(f"Prediction: Fake")

        # Display a more detailed prediction result
        st.write(f"Confidence Score: {prediction:.2f}")
else:
    st.info("Please upload an image file to get a prediction.")


##to run python -m streamlit run c:/Users/ABHIMANYU/Desktop/dataScience/deepfake_detection/app.py
