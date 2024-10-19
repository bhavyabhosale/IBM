import os
import json
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai

# Configure the API key for Google Generative AI
api_key = "AIzaSyCHilfAjm4gtEQG8proy3igpmoT1Rzx9iY"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Set page configuration
st.set_page_config(
    page_title="AgriAI", 
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@master/assets/72x72/1f33f.png", 
    layout='centered', 
    initial_sidebar_state="collapsed"
)

# Set the working directory and model paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names from the JSON file
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to get disease information using Google Generative AI
def get_disease_info(disease_name):
    prompt = f"Provide detailed information about the plant disease called '{disease_name}'."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Ensure you are using the correct model
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, 'text') else "No information available."
    except Exception as e:
        return f"Error fetching disease info: {e}"

# Load the machine learning model for crop recommendation
def load_model(modelfile):
    return pickle.load(open(modelfile, 'rb'))

# Sidebar options
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Crop Recommendation", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("Welcome to AgriAI 🌱")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(""" 
    **AgriAI** is an innovative platform designed to empower farmers and agricultural enthusiasts with cutting-edge AI technologies. Our mission is to enhance agricultural productivity and sustainability through intelligent decision-making tools.

    ### What We Offer:
    - **Crop Recommendation:** Our advanced algorithms analyze soil and environmental parameters to recommend the most suitable crops for your farm. This service helps optimize yields and minimize resource wastage.

    - **Plant Disease Recognition:** Upload an image of your plant leaves, and our AI will quickly identify any diseases present, providing actionable insights to help you maintain healthy crops and prevent losses.

    ### Key Features:
    - **User-Friendly Interface:** Our platform is designed to be intuitive, making it easy for users of all tech-savviness levels to navigate and utilize the tools available.

    - **Real-Time Insights:** Get immediate feedback on crop recommendations and disease identification, allowing you to take swift action in your farming practices.

    - **Data-Driven Decisions:** Leverage the power of data analytics to make informed decisions that can enhance your farming outcomes.

    - **Community Support:** Join a community of like-minded farmers and enthusiasts who share best practices, tips, and support each other in their agricultural journeys.

    ### Get Started:
    - **Crop Recommendation:** Select the "Crop Recommendation" option from the sidebar to input your soil and environmental data.
    - **Disease Recognition:** Go to "Disease Recognition" to upload an image of a plant leaf and receive immediate disease analysis.
    """)

# About Page
elif app_mode == "About":
    st.header("About AgriAI")
    st.markdown("""
    **AgriAI** is at the forefront of agricultural technology, using machine learning and artificial intelligence to support farmers in making data-informed decisions.

    ### Our Vision:
    We aim to revolutionize agriculture by integrating technology with traditional farming practices. By leveraging AI, we strive to create a future where farming is more efficient, sustainable, and accessible to all.

    ### Our Team:
    Our team consists of agricultural experts, data scientists, and software engineers who are passionate about using technology to solve real-world agricultural problems.

    ### How It Works:
    - **Crop Recommendation:** Our system evaluates a range of parameters, including soil health, nutrient levels, and climatic conditions. Using a comprehensive database of crop growth requirements, we provide tailored recommendations to optimize yields.

    - **Plant Disease Recognition:** Utilizing state-of-the-art image classification models, our AI analyzes uploaded images of plant leaves to detect diseases. Users receive a detailed report on the detected disease and suggested remedial actions.

    ### Why Choose AgriAI?
    - **Expertise in AI and Agriculture:** Our team comprises agricultural experts and data scientists who collaborate to build effective solutions.
    - **Continuous Learning:** Our models are continually updated with new data to enhance accuracy and reliability.
    - **Community Engagement:** We believe in empowering local farmers by providing educational resources and support through our platform.

    ### Join Us:
    Be part of the agricultural revolution! Whether you are a smallholder farmer or a large agricultural enterprise, AgriAI is here to support your journey toward a more productive and sustainable future.
    """)

# Crop Recommendation Page
elif app_mode == "Crop Recommendation":
    st.header("Crop Recommendation 🌱")
    st.subheader("Find the most suitable crop for your farm.")
    
    # User inputs for crop recommendation
    N = st.number_input("Nitrogen (N) - kg/ha", min_value=0, max_value=300, value=100)
    P = st.number_input("Phosphorus (P) - kg/ha", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K) - kg/ha", min_value=0, max_value=300, value=100)
    temp = st.number_input("Temperature (°C)", min_value=5, max_value=50, value=25)
    humidity = st.number_input("Humidity (%)", min_value=20, max_value=100, value=60)
    ph = st.number_input("pH Level", min_value=4.0, max_value=9.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=4000, value=1000)

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    label_mapping = {
        'apple': 0, 'banana': 1, 'blackgram': 2, 'chickpea': 3, 'coconut': 4, 'coffee': 5,
        'cotton': 6, 'grapes': 7, 'jute': 8, 'kidneybeans': 9, 'lentil': 10, 'maize': 11,
        'mango': 12, 'mothbeans': 13, 'mungbean': 14, 'muskmelon': 15, 'orange': 16,
        'papaya': 17, 'pigeonpeas': 18, 'pomegranate': 19, 'rice': 20, 'watermelon': 21
    }
    
    if st.button("Predict Crop"):
        loaded_model = load_model('model.pkl')
        prediction = loaded_model.predict(single_pred)
        predicted_label_index = prediction.item()
        predicted_label = next(key for key, value in label_mapping.items() if value == predicted_label_index)
        st.success(f"{predicted_label} is recommended for your farm.")

# Plant Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Plant Disease Recognition 🌿")
    uploaded_image = st.file_uploader("Upload an Image of a Plant Leaf:", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img, caption='Uploaded Image')

        with col2:
            if st.button("Classify"):
                prediction = predict_image_class(model, image, class_indices)
                st.success(f'Detected Disease: {str(prediction)}')
                
                # Fetch disease information
                disease_info = get_disease_info(prediction)
                st.markdown(f"### Disease Information:")
                st.write(disease_info)
