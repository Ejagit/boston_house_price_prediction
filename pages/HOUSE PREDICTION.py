import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn import datasets

# Function to load data from CSV
@st.cache_data
def load_data():
    return pd.read_csv('dataset/boston.csv')

# Function to load the trained model
@st.cache_data
def load_model():
    return joblib.load('model/ridge_reg_ten_model.joblib')

# Function to predict using the loaded model
def predict(data):
    model = load_model()
    return model.predict(data)

# Function to define user input features
def user_input_features(X):
    st.header('Input Features')
    crim = st.slider('crim', float(X.crim.min()), float(X.crim.max()), float(X.crim.mean()))
    zn = st.slider('zn', float(X.zn.min()), float(X.zn.max()), float(X.zn.mean()))
    indus = st.slider('indus', float(X.indus.min()), float(X.indus.max()), float(X.indus.mean()))
    chas = st.slider('chas', float(X.chas.min()), float(X.chas.max()), float(X.chas.mean()))
    nox = st.slider('nox', float(X.nox.min()), float(X.nox.max()), float(X.nox.mean()))
    rm = st.slider('rm', float(X.rm.min()), float(X.rm.max()), float(X.rm.mean()))
    tax = st.slider('tax', float(X.tax.min()), float(X.tax.max()), float(X.tax.mean()))
    ptratio = st.slider('ptratio', float(X.ptratio.min()), float(X.ptratio.max()), float(X.ptratio.mean()))
    black = st.slider('black', float(X.black.min()), float(X.black.max()), float(X.black.mean()))
    lstat = st.slider('lstat', float(X.lstat.min()), float(X.lstat.max()), float(X.lstat.mean()))
    
    data = {
        'crim': crim,
        'zn': zn,
        'indus': indus,
        'chas': chas,
        'nox': nox,
        'rm': rm,
        'tax': tax,
        'ptratio': ptratio,
        'black': black,
        'lstat': lstat
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Main Panel
st.title("Boston House Price Prediction")

# Load data
data = load_data()
X = data.drop(columns=['medv'])
y = data['medv']

# Display user input features
df = user_input_features(X)

# Display predictions when the Predict button is pressed
if st.button('Predict'):
    st.header('Prediction of House Price')
    predictions = predict(df)
    
    # Convert predicted medv to dollar and rupiah
    predicted_medv = predictions[0] * 1000  # Convert to thousand dollars
    predicted_rupiah = predicted_medv * 16350  # Assuming exchange rate 1 USD = 16350 IDR
    
    st.write(f'HOUSE PRICE IN DOLLAR: ${predicted_medv:.2f}')
    st.write(f'HOUSE PRICE IN RUPIAH: Rp {predicted_rupiah:.2f}')