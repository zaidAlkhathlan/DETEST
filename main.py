from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel


model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()


class InputFeatures(BaseModel):
    highest_value: int
    appearance: int
    minutes_played: int
    

def preprocessing(input_features: InputFeatures):
        """function that applies the same preprocessing steps (use
            d on the training data) to a new test row, ensuring consistenc
            y with the training data preprocessing pipeline."""
        dict_f = {
        'highest_value': input_features.highest_value,
        'appearance': input_features.appearance,
        'minutes_played': input_features.minutes_played,
          }
        
        # Convert dictionary values to a list in the correct order
        features_list = [dict_f[key] for key in sorted(dict_f)]
        # Scale the input features
        scaled_features = scaler.transform([list(dict_f.values())])

        return scaled_features

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/items/{item_id}")

# post request
@app.post("/items/{item_id}")
def create_item(item_id: int):
    
    return {"item": item_id}







