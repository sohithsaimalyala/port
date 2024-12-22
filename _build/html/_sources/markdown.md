# EAS503 Housing Price Prediction System

Objective
This project focuses on developing a Housing Price Prediction System using machine learning and deploying it as a scalable web application. The aim is to provide a real-time tool to predict housing prices based on various features such as property size, number of rooms, and location type.

## 1. Problem Definition and Objective
Define the problem statement: Predicting housing prices or classifications based on input features (e.g., size, rooms, location).
Build a containerized, deployable API for prediction and create a user-friendly Streamlit app for interaction.


## 2. Build the Machine Learning Model
Collect and preprocess the data (e.g., housing data).
Train the ML model using your chosen algorithm (e.g., linear regression, random forest).
Save the trained model using joblib or pickle.


## 3. Create the API with FastAPI
Define the API using FastAPI to serve predictions from the ML model.
Structure:
A GET endpoint for testing (/).
A POST endpoint for predictions (/predict).

## 4. Containerize the API with Docker
Write a Dockerfile:


## 5. Deploy the API
Deploy to a cloud platform:
DigitalOcean, Render, or AWS are common options.
Steps for DigitalOcean:
Create a Droplet (Ubuntu-based).
SSH into the Droplet.
Install Docker
Pull the container image from Docker Hub:
Run the container:

## 6. Create a Streamlit App
Write a Streamlit app to interact with the API.

## 7. Integrate MLFlow for Experiment Tracking
Use MLFlow or DagsHub to track your experiments and hyperparameters.

## 8. Create a JupyterBook Website

## 9.Final Testing and Debugging
Test all components:
  API.
  Streamlit app.
  JupyterBook links.
  Verify cross-platform compatibility.
  Collect feedback and iterate.
