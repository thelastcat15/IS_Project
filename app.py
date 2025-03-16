import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

def predict_sales(model, day, month, year, weekday, temperature, fuel_price, cpi, unemployment):
    input_data = np.array([[day, month, year, weekday, temperature, fuel_price, cpi, unemployment]])
    prediction = model.predict(input_data)
    return prediction[0]

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image(img, model):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    class_names = ['cloudy', 'rainy', 'shine', 'sunrise']
    return class_names[predicted_class_index]

# โหลดโมเดลที่ฝึกไว้
rf_model = joblib.load("random_forest_model.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
nn_model = load_model("nn_model.h5")

# UI ของแอป Streamlit
st.sidebar.title("เมนูนำทาง")
page = st.sidebar.radio("ไปที่", ["หน้าแรก", "ข้อมูล & โมเดล", "ทดสอบ Random Forest", "ทดสอบ Linear Regression", "ทดสอบ Neural Network"])

if page == "หน้าแรก":
    st.title("โครงการทำนายยอดขาย & จำแนกภาพ")
    st.write("โครงการนี้ใช้ Machine Learning และ Neural Network เพื่อทำนายยอดขายและจำแนกภาพสภาพอากาศ")

elif page == "ข้อมูล & โมเดล":
    st.title("ข้อมูล & คำอธิบายโมเดล")
    st.write("### ชุดข้อมูล")
    st.write("เราใช้ชุดข้อมูลยอดขายของร้านค้า และภาพสภาพอากาศ")
    st.write("### โมเดลที่ใช้")
    st.write("1. Random Forest Regressor - ทำนายยอดขายจากปัจจัยเศรษฐกิจ")
    st.write("2. Linear Regression - ใช้โมเดลเชิงเส้นในการทำนายยอดขาย")
    st.write("3. Neural Network - ใช้ CNN ในการจำแนกประเภทสภาพอากาศ")

elif page == "ทดสอบ Random Forest":
    st.title("ทำนายยอดขายด้วย Random Forest")
    day = st.number_input("วัน", min_value=1, max_value=31, value=1)
    month = st.number_input("เดือน", min_value=1, max_value=12, value=1)
    year = st.number_input("ปี", min_value=2000, max_value=2030, value=2023)
    weekday = st.number_input("วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)", min_value=0, max_value=6, value=0)
    temperature = st.number_input("อุณหภูมิ", value=25.0)
    fuel_price = st.number_input("ราคาน้ำมัน", value=3.5)
    cpi = st.number_input("CPI", value=200.0)
    unemployment = st.number_input("อัตราการว่างงาน", value=5.0)
    
    if st.button("ทำนายยอดขาย"):
        prediction = predict_sales(rf_model, day, month, year, weekday, temperature, fuel_price, cpi, unemployment)
        st.write(f"### คาดการณ์ยอดขาย: ${prediction:,.2f}")

elif page == "ทดสอบ Linear Regression":
    st.title("ทำนายยอดขายด้วย Linear Regression")
    day = st.number_input("วัน", min_value=1, max_value=31, value=1)
    month = st.number_input("เดือน", min_value=1, max_value=12, value=1)
    year = st.number_input("ปี", min_value=2000, max_value=2030, value=2023)
    weekday = st.number_input("วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)", min_value=0, max_value=6, value=0)
    temperature = st.number_input("อุณหภูมิ", value=25.0)
    fuel_price = st.number_input("ราคาน้ำมัน", value=3.5)
    cpi = st.number_input("CPI", value=200.0)
    unemployment = st.number_input("อัตราการว่างงาน", value=5.0)
    
    if st.button("ทำนายยอดขาย"):
        prediction = predict_sales(lr_model, day, month, year, weekday, temperature, fuel_price, cpi, unemployment)
        st.write(f"### คาดการณ์ยอดขาย: ${prediction:,.2f}")

elif page == "ทดสอบ Neural Network":
    st.title("ทดสอบโมเดล Neural Network")
    uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)
        prediction = predict_image(img, nn_model)
        st.write(f"### คลาสที่คาดการณ์: {prediction}")
