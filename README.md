# Boston House Price Prediction

This project demonstrates the end-to-end process of building a machine learning model to predict house prices in Boston, deploying it with Streamlit, and using a dataset sourced from Kaggle.

## Overview

The goal of this project is to develop a regression model that predicts house prices based on various features such as crime rate, zoning, and others provided in the dataset. The model is trained on historical data from Boston.

## Dataset

The dataset used in this project is sourced from Kaggle's Boston Housing dataset. It includes various features such as crime rate, residential land zone, nitric oxides concentration, number of rooms, property tax rate, pupil-teacher ratio, and more.

## Machine Learning Model

### Model Selection

The model chosen for this project is Ridge Regression. Ridge Regression is a linear regression model that incorporates regularization to prevent overfitting and improve generalization.

### Model Training and Evaluation

The model is trained using the historical Boston housing dataset. Cross-validation techniques are employed to evaluate and fine-tune the model parameters for better performance.

### Model Performance Metrics

The performance of the model is evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score to assess its accuracy in predicting house prices.

## Deployment with Streamlit

### Streamlit App

The trained Ridge Regression model is deployed using Streamlit, a Python library for creating web applications. Users can interact with the app to input values for various features and get real-time predictions of house prices.

### How to Run the App

To run the Streamlit app locally:
1. Clone this repository.
2. Install the required Python packages (`streamlit`, `pandas`, `scikit-learn`, etc.).
3. Navigate to the project directory in your terminal.
4. Run the following command
