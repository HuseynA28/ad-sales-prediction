from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

import os
import joblib

# Print the current working directory to verify the path
print("Current working directory:", os.getcwd())

# Use an absolute path for debugging purposes to ensure the model can be loaded correctly
import os
import joblib

# Full, correct path to the model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'sales_model.pkl')
print("Full path to model:", model_path)

model = joblib.load(model_path)


# Load your trained model
model = joblib.load(model_path)


app = FastAPI()

# Define a Pydantic model for the input data
class Advertising(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

@app.post("/predict/")
async def predict(request: Advertising):
    data = np.array([[request.TV, request.Radio, request.Newspaper]])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}

@app.get("/")
async def read_root():
    return {"Hello": "World"}
