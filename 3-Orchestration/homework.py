import os
import pandas as pd
import pickle
from urllib.request import urlretrieve

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

import datetime
from dateutil.relativedelta import relativedelta

@task(name = "Get Path")
def get_paths(date):
    logger = get_run_logger()
    if date is None:
        dt_obj = datetime.date.today()
        logger.warning("Date is None, today's date will be used")
    else:
        dt_obj = datetime.datetime.strptime(date, "%Y-%m-%d")

    
    files = os.listdir('./data')
    
    train_date = dt_obj - relativedelta(months=2)
    val_date = dt_obj - relativedelta(months=1)
    
    train_file = f"fhv_tripdata_{str(train_date.year).zfill(2)}-{str(train_date.month).zfill(2)}.parquet"
    val_file = f"fhv_tripdata_{str(val_date.year).zfill(2)}-{str(val_date.month).zfill(2)}.parquet"

    for file in [train_file, val_file]:
        if file not in files:
            logger.warning(f"Files couldn't found : {file}")
            try:
                urlretrieve(f'https://nyc-tlc.s3.amazonaws.com/trip+data/{file}', filename = f'./data/{file}')
            except:
                logger.error(f"Error when downloading file : {file}")
        else:
            logger.info(f"File found : {file}")
    
    return f'./data/{train_file}', f'./data/{val_file}'

@task(name = "Read Data")
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task(name = "Prepare Features")
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task(name = "Model Train")
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    with mlflow.start_run():
        mlflow.set_tag("model", "LinearRegression")
        mlflow.set_tag("mode", "Training")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred, squared=False)
        logger.info(f"The MSE of training is: {mse}")
        mlflow.log_metric("mse", mse)

    return lr, dv

@task(name = "Run Model")
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    with mlflow.start_run():
        mlflow.set_tag("model", "LinearRegression")
        mlflow.set_tag("mode", "Validation")
        y_pred = lr.predict(X_val)
        y_val = df.duration.values

        mse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("mse", mse)
        logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner = SequentialTaskRunner)
def main(date = None):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("orch_homework")
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    with open(f"preprocessors/dv-{date}.b", "w+b") as preprocessor:
            pickle.dump(dv, preprocessor)
    with open(f"models/model-{date}.bin", "w+b") as model:
            pickle.dump(lr, model)

    run_model(df_val_processed, categorical, dv, lr)

DeploymentSpec(
    name="model_training_hw",
    flow=main,
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *"),
    tags=['hw']
)