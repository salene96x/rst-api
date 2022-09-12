from fastapi import FastAPI
from models import *
import joblib
import pandas as pd
import json

app = FastAPI()

@app.post("/get_predicted_price/")
async def get_predicted_price(data: Data):
    model = get_model()
    features = to_features(data)
    predicted_sale_price = model.predict(features)
    body = {
        "Response" : {
            "PredictedSalePrice" : {predicted_sale_price}
        }
    }
    return body
    
def to_features(data):
    data = json.dumps(data.__dict__)
    return pd.DataFrame.from_dict(data)

def get_model():
    return joblib.load('model.pkl')