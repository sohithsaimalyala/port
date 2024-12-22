# Final Code

This section contains the final code used in this project, including the FastAPI app, Streamlit app, and Dockerfile.

---

## FastAPI Code

```python
# Paste the FastAPI code here
# Final Code

This section contains the final code used in this project, including the FastAPI app, Streamlit app, and Dockerfile.

---

## FastAPI Code

```python
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Input model for the prediction endpoint
class InputModel(BaseModel):
    size: float  # Property size in square footage
    rooms: int   # Number of rooms
    location: str  # Location type: Urban, Suburban, Rural

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputModel):
    # Simple prediction logic
    prediction = data.size * 1000 + data.rooms * 500
    return {
        "input": {
            "size": data.size,
            "rooms": data.rooms,
            "location": data.location
        },
        "prediction": prediction
    }



# Paste the Streamlit app code here
import streamlit as st
import requests

# Backend FastAPI URL
API_ENDPOINT = "http://your-fastapi-url/predict"  # Replace with actual FastAPI endpoint

# Streamlit app title and description
st.title("Real-Time Housing Price Predictions")
st.write("Enter property details to get a price prediction.")

# Sidebar inputs
size = st.sidebar.number_input("Property Size (Square Footage)", value=1000.0)
rooms = st.sidebar.number_input("Number of Rooms", value=3, min_value=1)
location = st.sidebar.selectbox("Location Type", ["Urban", "Suburban", "Rural"])

# Input data dictionary
input_data = {
    "size": size,
    "rooms": rooms,
    "location": location
}

# Display user inputs
st.write("### Your Inputs")
st.json(input_data)

# Button to get prediction
if st.button("Get Prediction"):
    try:
        # POST request to FastAPI
        response = requests.post(API_ENDPOINT, json=input_data)
        response.raise_for_status()
        result = response.json()

        # Display the prediction
        st.success(f"Predicted Price: ${result['prediction']}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")



# Paste the Dockerfile code here
# Use the official Python image as the base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]



---

Once done, you can rebuild the JupyterBook using `jupyter-book build ./` from the project root directory, and the `code.md` content will appear on the website under the appropriate section.
