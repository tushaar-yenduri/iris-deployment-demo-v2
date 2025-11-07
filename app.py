import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🌸 Iris Flower Species Prediction App")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    # Mapping numeric label to class name
    label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    predicted_species = label_map[prediction]

    st.success(f"The predicted Iris species is: **{predicted_species}** 🌼")
