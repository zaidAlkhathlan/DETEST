from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import numpy as np


model = joblib.load('Kmeanes.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()

class InputFeatures(BaseModel):
    highest_value: int
    appearance: int
    minutes_played: int
    award: int

def preprocessing(input_features: InputFeatures):
        """function that applies the same preprocessing steps (use
            d on the training data) to a new test row, ensuring consistenc
            y with the training data preprocessing pipeline."""
        dict_f = {
        'highest_value': input_features.highest_value,
        'appearance': input_features.appearance,
        'minutes_played': input_features.minutes_played,
        'award': input_features.award,
        
          }
        
        # Convert dictionary values to a list in the correct order
        features_list = [dict_f[key] for key in sorted(dict_f)]
        # Scale the input features
        scaled_features = scaler.transform([list(dict_f.values())])

        return scaled_features

@app.post('/predict')
def predict(data: dict):
    print(f"Received Data: {data}")  # Check if API receives correctly
    input_data = np.array([[data['appearance'], data['minutes_played'], data['award'], data['highest_value']]])
    print(f"Before Scaling: {input_data}")

    scaled_input = scaler.transform(input_data)
    print(f"Scaled Input Features: {scaled_input}")

    prediction = model.predict(scaled_input)
    print(f"Prediction: {prediction}")

    return {"prediction": prediction.tolist()}


# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/items/{item_id}")

# post request
@app.post("/items/{item_id}")
def create_item(item_id: int):
    
    return {"item": item_id}







