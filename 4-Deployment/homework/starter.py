#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import sys
import numpy as np

def get_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    return dv, lr


def process_data(df):
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')

    return dicts

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    return df

def get_data(year, month):
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
    dicts = process_data(df)
    return dicts

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    dv, lr = get_model()
    dicts = get_data(year, month)
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(np.mean(y_pred))


if __name__ == "__main__":
    run()