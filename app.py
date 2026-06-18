import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_model.pkl")

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈"
)
st.sidebar.title("Project Information")

st.sidebar.info("""
Advanced Multi-Variable Sales Forecasting System

Machine Learning Model:
Random Forest Regressor

Dataset:
Walmart Sales Dataset

Developed using:
Python, Pandas, Scikit-Learn, Streamlit
""")

st.title("📈 Walmart Sales Forecasting System")

st.write("Enter details below to predict weekly sales.")

store = st.number_input("Store Number", min_value=1)

holiday = st.selectbox(
    "Holiday Flag",
    [0, 1]
)

temperature = st.number_input(
    "Temperature",
    value=25.0
)

fuel_price = st.number_input(
    "Fuel Price",
    value=3.0
)

cpi = st.number_input(
    "CPI",
    value=200.0
)

unemployment = st.number_input(
    "Unemployment",
    value=7.0
)

year = st.number_input(
    "Year",
    value=2012
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1
)

week = st.number_input(
    "Week",
    min_value=1,
    max_value=53,
    value=1
)

if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        'Store': [store],
        'Holiday_Flag': [holiday],
        'Temperature': [temperature],
        'Fuel_Price': [fuel_price],
        'CPI': [cpi],
        'Unemployment': [unemployment],
        'Year': [year],
        'Month': [month],
        'Week': [week]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Weekly Sales: ${prediction[0]:,.2f}"
    )
    