import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set page config for better visuals
st.set_page_config(page_title="House Price Prediction", layout="centered")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("housing.csv")
    data.dropna(inplace=True)
    return data

data = load_data()

# Display title and description
st.title("🏡 House Price Prediction")
st.write("This app predicts house prices based on various features using a Linear Regression model.")



# Prepare data for training
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# One-hot encoding for categorical variables
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Train the linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predict function
def predict_price(input_data):
    input_data = pd.DataFrame(input_data, index=[0])
    input_data = pd.get_dummies(input_data)
    missing_cols = set(x_train.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[x_train.columns]  # Ensure same order of columns
    prediction = reg.predict(input_data)
    return prediction[0]



# User input for predictions
# st.title("🏡 House Price Prediction")
# st.markdown("### Enter the details below to predict the house price:")

longitude = st.number_input("📍 Longitude", value=data['longitude'].mean())
latitude = st.number_input("📍 Latitude", value=data['latitude'].mean())
housing_median_age = st.number_input("🏠 Housing Median Age", value=data['housing_median_age'].mean())
total_rooms = st.number_input("🛏️ Total Rooms", value=data['total_rooms'].mean())
total_bedrooms = st.number_input("🛌 Total Bedrooms", value=data['total_bedrooms'].mean())
population = st.number_input("👥 Population", value=data['population'].mean())
households = st.number_input("🏘️ Households", value=data['households'].mean())
median_income = st.number_input("💵 Median Income", value=data['median_income'].mean())

ocean_proximity = st.selectbox("🌊 Ocean Proximity", data['ocean_proximity'].unique())

# Create a dictionary for the input
input_data = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_proximity
}

# # Show prediction
# if st.button("Predict"):
#     price = predict_price(input_data)
#     st.write(f"Predicted House Price: ${price:.2f}")

# Show prediction
if st.button("💡 Predict House Price"):
    price = predict_price(input_data)
    st.success(f"🏠 Predicted House Price: **${price:.2f}**")

# Model performance
st.subheader("Model Performance")
score = reg.score(x_test, y_test)
st.write(f"R-squared score of the model: {score:.4f}")
 
