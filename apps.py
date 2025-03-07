import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("D:\\college\\Laptop-Price-Predictor-main\\Laptop-Price-Predictor-main\\pipe.pkl", "rb"))

# Function for prediction
def price_predictor(Company, typename, ram, weight, touchscreen, ips, screensize, screen_res, cpu_brand, hdd, ssd, gpu_brand, os):
    X_res, Y_res = map(int, screen_res.lower().replace(" ", "").split('x'))
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screensize
    
    # Convert input to numpy array
    query = np.array([Company, typename, ram, weight, touchscreen, ips, ppi, cpu_brand, hdd, ssd, gpu_brand, os], dtype=object).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(query)
    predicted_price = np.exp(prediction[0])  # Apply exponent if model was trained with log prices
    
    return round(predicted_price)

# Streamlit GUI
st.title("💻 Laptop Price Predictor")
st.write("Enter laptop specifications to estimate the price.")

# User inputs
Company = st.selectbox("Brand", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"])
typename = st.selectbox("Type", ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible"])
ram = st.slider("RAM (GB)", 4, 64, step=4)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.radio("Touchscreen", [0, 1])
ips = st.radio("IPS Display", [0, 1])
screensize = st.slider("Screen Size (inches)", 11.0, 18.0, step=0.1)
screen_res = st.text_input("Screen Resolution (e.g., 1920x1080)", "1920x1080")
cpu_brand = st.selectbox("Processor", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Ryzen 5", "AMD Ryzen 7"])
hdd = st.number_input("HDD (GB)", min_value=0, max_value=2000, step=128)
ssd = st.number_input("SSD (GB)", min_value=0, max_value=2000, step=128)
gpu_brand = st.selectbox("GPU Brand", ["Intel", "AMD", "Nvidia"])
os = st.selectbox("Operating System", ["Windows", "MacOS", "Linux", "Other"])

# Predict button
if st.button("Predict Price 💰"):
    result = price_predictor(Company, typename, ram, weight, touchscreen, ips, screensize, screen_res, cpu_brand, hdd, ssd, gpu_brand, os)
    st.success(f"Estimated Laptop Price: ₹ {result}")

