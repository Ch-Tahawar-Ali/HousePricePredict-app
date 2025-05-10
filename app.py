import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load

# Load your existing code with minor modifications
# -------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

# Load data and model
@st.cache_data
def load_data():
    data = pd.read_csv("housing.csv")
    return data

@st.cache_resource
def load_model():
    return load("model_pipeline.joblib")

data = load_data()
model = load_model()

# Split data for demonstration
x = data.drop(columns=['median_house_value'])
y = data['median_house_value']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
y_pred = model.predict(x_test)

# Streamlit app interface
# -------------------------------------------------
st.title("California Housing Price Predictor 🏠")

st.write("""
This app predicts median house values in California using machine learning!
""")

# Sidebar for user input
st.sidebar.header("User Input Features")

def user_input_features():
    longitude = st.sidebar.number_input("Longitude", value=-122.23)
    latitude = st.sidebar.number_input("Latitude", value=37.88)
    housing_median_age = st.sidebar.number_input("Housing Median Age", min_value=1, value=41)
    total_rooms = st.sidebar.number_input("Total Rooms", min_value=1, value=880)
    total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1, value=129)
    population = st.sidebar.number_input("Population", min_value=1, value=322)
    households = st.sidebar.number_input("Households", min_value=1, value=126)
    median_income = st.sidebar.number_input("Median Income", min_value=0.0, value=8.3252)
    ocean_proximity = st.sidebar.selectbox("Ocean Proximity", options=[
        'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
    
    return pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })

input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Prediction and results
if st.sidebar.button("Predict"):
    prediction = model.predict(input_df)
    
    st.subheader("Prediction")
    st.success(f"Predicted Median House Value: ${prediction[0]:,.2f}")
    
    # Model performance
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
    with col2:
        st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    
    # Visualization
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted House Values")
    st.pyplot(fig)
    
    # Feature distribution
    st.subheader("Feature Distributions")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    data['median_income'].hist(ax=ax2)
    ax2.set_title("Median Income Distribution")
    st.pyplot(fig2)

# Data exploration section
if st.checkbox("Show Raw Data"):
    st.subheader("Housing Data")
    st.write(data)

if st.checkbox("Show Data Statistics"):
    st.subheader("Data Statistics")
    st.write(data.describe())