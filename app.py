import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

# loading the model
model = pickle.load(open('car.pkl', 'rb'))

# Title of the app
st.title("Car Selling Price Prediction")

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    # car_name = st.text_input("Car Name")
    # present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, max_value=50.0, step=0.1)
    Present_Price = st.number_input("show room price(in lakhs)" , 0 , )
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, step=100)
    fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox("Seller Type", options=['Dealer', 'Individual'])

with col2:
    transmission = st.selectbox("Transmission", options=['Manual', 'Automatic'])
    owner = st.selectbox("Owner", options=[0, 1, 2, 3])
    age = st.number_input("Age of the car (in years)", min_value=0, max_value=50, step=1)

# Initialize LabelEncoders for categorical features
le_fuel_type = LabelEncoder().fit(['Petrol', 'Diesel', 'CNG'])
le_seller_type = LabelEncoder().fit(['Dealer', 'Individual'])
le_transmission = LabelEncoder().fit(['Manual', 'Automatic'])

# Transform categorical inputs
fuel_type_encoded = le_fuel_type.transform([fuel_type])[0]
seller_type_encoded = le_seller_type.transform([seller_type])[0]
transmission_encoded = le_transmission.transform([transmission])[0]

# Predict button
if st.button("Predict Selling Price"):
    # Create input array for prediction
    input_data = np.array([[kms_driven, fuel_type_encoded, seller_type_encoded, 
                            transmission_encoded, owner, age]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.success(f"The predicted selling price of the car ' is: ₹{prediction:.2f} lakhs")
