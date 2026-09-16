import streamlit as st
import requests
st.set_page_config(page_title="Breast Cancer Prediction",
                   page_icon="🩺",
                   layout="wide")
st.title("🩺 Breast Cancer Prediction")
st.write("Enter the tumor measurements below to get a model prediction.")
API_URL = "http://127.0.0.1:8000/predict"
features = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
          "smoothness_mean", "compactness_mean", "concavity_mean",
          "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
          "radius_se", "texture_se", "perimeter_se", "area_se",
          "smoothness_se", "compactness_se", "concavity_se",
          "concave_points_se", "symmetry_se", "fractal_dimension_se",
          "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
          "smoothness_worst", "compactness_worst", "concavity_worst",
          "concave_points_worst", "symmetry_worst",
          "fractal_dimension_worst", "radius_perimeter_ratio"]
input_data = {}
st.subheader("Tumor Measurements")
for feature in features:
    input_data[feature] = st.number_input(feature,min_value=0.0,format="%.6f")
if st.button("Predict"):
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()

            st.success(f"Model Prediction: {result['result']}")
            st.write(f"Prediction value: {result['prediction']}")

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.RequestException:
        st.error("Could not connect to the FastAPI backend.")
