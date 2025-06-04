import streamlit  as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import r2_score

# Load the preprocessor and model
preprocessor = joblib.load("preprocessor.joblib")
model = joblib.load("model.joblib")

# App title
st.title("🏡 Housing Price Prediction App")
st.write("This app predicts house prices based on various features using a trained Random Forest model.")

# Define user inputs
st.sidebar.header("Enter House Details")

def user_input():
    area = st.sidebar.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
    stories = st.sidebar.slider("Stories", 1, 4, 2)
    parking = st.sidebar.slider("Parking Spots", 0, 5, 1)
    
    mainroad = st.sidebar.selectbox("Main Road Access", ["yes", "no"])
    guestroom = st.sidebar.selectbox("Guest Room", ["yes", "no"])
    basement = st.sidebar.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.sidebar.selectbox("Air Conditioning", ["yes", "no"])
    prefarea = st.sidebar.selectbox("Preferred Area", ["yes", "no"])
    furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    return pd.DataFrame([data])

# Get input data
input_df = user_input()

# Predict button
if st.button("Predict Price"):
    # Preprocess and predict
    input_transformed = preprocessor.transform(input_df)
    prediction = model.predict(input_transformed)[0]
    st.subheader(f"🏷️ Predicted House Price: **{round(prediction, 2):,} PKR**")

# Display model accuracy
if st.checkbox("Show Model Accuracy (R² on training data)"):
    df = pd.read_csv("Housing.csv")
    X = df.drop("price", axis=1)
    y = df["price"]
    X_preprocessed = preprocessor.transform(X)
    y_pred = model.predict(X_preprocessed)
    r2 = r2_score(y, y_pred)
    st.write(f"📊 R² Score on Training Data: **{r2:.4f}**")
