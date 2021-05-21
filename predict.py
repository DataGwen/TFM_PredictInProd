import os
from math import sqrt

import joblib
import pandas as pd
from TaxiFareModel.params import *
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import pytz


def get_test_data(nrows, data="s3"):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

    if data == "local":
        df = pd.read_csv(path)
    elif data == "full":
        df = pd.read_csv(AWS_BUCKET_TEST_PATH)
    else:
        df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
    return df


def download_model(model_directory="PipelineTest", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(nrows, kaggle_upload=False):
    df_test = get_test_data(nrows)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)
        
def generate_prediction(df_pred):
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_pred)
    else:
        y_pred = pipeline.predict(df_pred)
    return  y_pred
        
def generate_prediction_from_api(pickup_datetime,
                                 pickup_longitude,
                                 pickup_latitude,
                                 dropoff_longitude,
                                 dropoff_latitude,
                                 passenger_count):
    
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")


    data = [['2013-07-06 17:18:00.000000119', 
             formatted_pickup_datetime, 
             float(pickup_longitude),
             float(pickup_latitude),
             float(dropoff_longitude),
             float(dropoff_latitude),
             int(passenger_count)]]
                         
    X_pred = pd.DataFrame(data, columns=['key', 
                                         'pickup_datetime',
                                         'pickup_longitude', 
                                         'pickup_latitude', 
                                         'dropoff_longitude', 
                                         'dropoff_latitude',
                                         'passenger_count'])
    print(X_pred)
    y_pred = generate_prediction(X_pred)
    return y_pred

if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    # nrows = 100
    # generate_submission_csv(nrows, kaggle_upload=False)
    y_pred = generate_prediction_from_api("2013-07-06 17:18:00", "-73.950655", "40.783282", "-73.984365","40.769802", "1")
    print(y_pred)