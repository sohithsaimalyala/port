# Final Code

This is the final code for the project:

```{code-block} python
:linenos:

from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Define the input model
class YourInputModel(BaseModel):
    size: float
    rooms: int
    location: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI"}

# Predict endpoint
@app.post("/predict")
def predict(data: YourInputModel):
    prediction = (data.size * 1000) + (data.rooms * 500)
    return {"input": data.dict(), "prediction": prediction}
