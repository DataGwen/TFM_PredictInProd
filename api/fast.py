
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(
    pickup_datetime,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    passenger_count):
    
    y_pred = generate_prediction_from_api(pickup_datetime,
                                        pickup_longitude,
                                        pickup_latitude,
                                        dropoff_longitude,
                                        dropoff_latitude,
                                        passenger_count)

    return {"prediction": str(y_pred[0])}


