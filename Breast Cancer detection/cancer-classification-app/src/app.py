import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model  # Import for loading .keras model
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = load_model(r'c:\Users\admin\Desktop\Apps\Resume and Shit\Portfolio\Projects\ML Apps\ML_Projects\Breast Cancer detection\cancer-classification-app\src\models\cancer_nn_model.keras')
scaler = joblib.load(r'c:\Users\admin\Desktop\Apps\Resume and Shit\Portfolio\Projects\ML Apps\ML_Projects\Breast Cancer detection\cancer-classification-app\src\models\scaler.pkl')  # Load the pre-fitted scaler

# Function to preprocess input data
def preprocess_input(data):
    data_scaled = scaler.transform(data)  # Use transform instead of fit_transform
    return data_scaled

# Streamlit app
st.title("Breast Cancer Classification")

# User input for features
st.sidebar.header("User Input Features")
def user_input_features():
    mean_radius = st.sidebar.number_input("Mean Radius", min_value=0.0)
    mean_texture = st.sidebar.number_input("Mean Texture", min_value=0.0)
    mean_perimeter = st.sidebar.number_input("Mean Perimeter", min_value=0.0)
    mean_area = st.sidebar.number_input("Mean Area", min_value=0.0)
    mean_smoothness = st.sidebar.number_input("Mean Smoothness", min_value=0.0)
    
    # Add more features as needed
    features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    return features

input_data = user_input_features()

# Preprocess the input data
input_data_scaled = preprocess_input(input_data)

# Make predictions
prediction = model.predict(input_data_scaled)
prediction_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

# Display results
st.subheader("Prediction")
if prediction_class[0] == 0:
    st.write("The model predicts: Benign")
else:
    st.write("The model predicts: Malignant")

st.subheader("Prediction Probability")
st.write(f"Benign: {prediction[0][0]:.2f}")
st.write(f"Malignant: {prediction[0][1]:.2f}")