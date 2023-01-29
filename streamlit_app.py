import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sodapy import Socrata
import plotly.express as px

st.set_page_config(
    page_title='New York City Taxi Prediction'
)

st.title('New York City Taxi Prediction')


class IngestData:

    def __init__(self, files_green=None, files_yellow=None, yellow_api="m6nq-qud6"):
        self.files_green = files_green
        self.files_yellow = files_yellow
        self.green_taxi_data = None
        self.yellow_taxi_data = None
        self.green_api = None
        self.yellow_api = None

    # def read_green_taxi_from_api(self, limit):
    #     client = Socrata("data.cityofnewyork.us", None)
    #     results = client.get(self.green_api, limit=limit)
    #     self.green_taxi_data= pd.DataFrame.from_records(results)   

    def read_yellow_taxi_from_api(self, limit):
        client = Socrata("data.cityofnewyork.us", None)
        results = client.get("m6nq-qud6", limit=limit)
        self.yellow_taxi_data= pd.DataFrame.from_records(results)  


    def read_data_green(self, files):
        self.green_taxi_data = pd.concat([pd.read_parquet(file) for file in files])
        

    def read_data_yellow(self, files):
        self.yellow_taxi_data = pd.concat([pd.read_parquet(file) for file in files])

    def fill_na(self):
        self.yellow_taxi_data.passenger_count.fillna(self.yellow_taxi_data.passenger_count.median(), inplace=True)

    # def change_green_types(self):
    #     self.green_taxi_data['PULocationID'] = self.green_taxi_data['pulocationid'].astype('int64')
    #     self.green_taxi_data['DOLocationID'] = self.green_taxi_data['dolocationid'].astype('int64')
    #     self.green_taxi_data['fare_amount'] = self.green_taxi_data['fare_amount'].astype('float64')
    #     self.green_taxi_data['total_amount'] = self.green_taxi_data['total_amount'].astype('float64')
    #     self.green_taxi_data['passenger_count'] = self.green_taxi_data['passenger_count'].astype('int64')
    #     self.green_taxi_data['trip_distance'] = self.green_taxi_data['trip_distance'].astype('float64')

    def change_yellow_types(self):
        self.yellow_taxi_data['PULocationID'] = self.yellow_taxi_data['pulocationid'].astype('int64')
        self.yellow_taxi_data['DOLocationID'] = self.yellow_taxi_data['dolocationid'].astype('int64')
        self.yellow_taxi_data['fare_amount'] = self.yellow_taxi_data['fare_amount'].astype('float64')
        self.yellow_taxi_data['total_amount'] = self.yellow_taxi_data['total_amount'].astype('float64')
        self.yellow_taxi_data['passenger_count'] = self.yellow_taxi_data['passenger_count'].astype('int64')
        self.yellow_taxi_data['trip_distance'] = self.yellow_taxi_data['trip_distance'].astype('float64')

        
        
    def create_target(self):
            self.yellow_taxi_data['tpep_pickup_datetime'] = pd.to_datetime(self.yellow_taxi_data['tpep_pickup_datetime'])
            self.yellow_taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(self.yellow_taxi_data['tpep_dropoff_datetime'])

            self.yellow_taxi_data['trip_duration'] = self.yellow_taxi_data['tpep_dropoff_datetime'] - self.yellow_taxi_data['tpep_pickup_datetime']
            self.yellow_taxi_data['trip_duration'] = self.yellow_taxi_data['trip_duration'].dt.total_seconds()

            # self.green_taxi_data['lpep_pickup_datetime'] = pd.to_datetime(self.green_taxi_data['lpep_pickup_datetime'])
            # self.green_taxi_data['lpep_dropoff_datetime'] = pd.to_datetime(self.green_taxi_data['lpep_dropoff_datetime'])

            # self.green_taxi_data['trip_duration'] = self.green_taxi_data['lpep_dropoff_datetime'] - self.green_taxi_data['lpep_pickup_datetime']
            # self.green_taxi_data['trip_duration'] = self.green_taxi_data['trip_duration'].dt.total_seconds()

    # def dropping_cols(self):
        # cols_to_keep = ['trip_distance', 'passenger_count', 'trip_duration', "store_and_fwd_flag", "VendorID"]

        # cols_to_drop_yellow = [col for col in self.yellow_taxi_data.columns if col not in cols_to_keep]
        # self.yellow_taxi_data.drop(columns=cols_to_drop_yellow, axis =1,  inplace=True)

        # cols_to_drop_green = [col for col in self.yellow_taxi_data.columns if col not in cols_to_keep]
        # self.green_taxi_data.drop(columns=cols_to_drop_green, axis=1,  inplace=True)

        # self.yellow_taxi_data = self.yellow_taxi_data[['trip_distance', 'passenger_count', 'trip_duration', "store_and_fwd_flag", "VendorID"]]
        # self.green_taxi_data = self.green_taxi_data[['trip_distance', 'passenger_count', 'trip_duration', "store_and_fwd_flag", "VendorID"]]

        
    def dup_and_miss(self):
        # print(f"Number of duplicated rows in yellow taxi data: {self.yellow_taxi_data.duplicated().sum()}")
        # print(f"Number of NA rows in yellow taxi data: {self.yellow_taxi_data.isna().sum().sum()}")
        print(f"Number of duplicated rows in green taxi data: {self.yellow_taxi_data.duplicated().sum()}")
        print(f"Number of NA rows in green taxi data: {self.yellow_taxi_data.isna().sum().sum()}")


    def outlier_removal(self):
        self.yellow_taxi_data = self.yellow_taxi_data[(self.yellow_taxi_data.trip_duration < 5600)]
        self.yellow_taxi_data = self.yellow_taxi_data[(self.yellow_taxi_data.trip_duration > 0)]
        self.yellow_taxi_data = self.yellow_taxi_data[(self.yellow_taxi_data.passenger_count > 0)]
        self.yellow_taxi_data = self.yellow_taxi_data[(self.yellow_taxi_data.trip_distance < 50000)]
        self.yellow_taxi_data = self.yellow_taxi_data[(self.yellow_taxi_data.fare_amount < 50000)]
        self.yellow_taxi_data = self.yellow_taxi_data[(self.yellow_taxi_data.total_amount < 50000)]




        # self.green_taxi_data = self.green_taxi_data[(self.green_taxi_data.trip_duration < 5600)]
        # self.green_taxi_data = self.green_taxi_data[(self.green_taxi_data.trip_duration > 0)]
        # self.green_taxi_data = self.green_taxi_data[(self.green_taxi_data.passenger_count > 0)]
        # self.green_taxi_data = self.green_taxi_data[(self.green_taxi_data.trip_distance < 50000)]
        # self.green_taxi_data = self.green_taxi_data[(self.green_taxi_data.fare_amount < 50000)]
        # self.green_taxi_data = self.green_taxi_data[(self.green_taxi_data.total_amount < 50000)]
ingest = IngestData()
ingest.read_yellow_taxi_from_api(1000)

st.write("Here is an example of NYC yellow taxi data taken from the New York City Open Data API.")
st.write(ingest.yellow_taxi_data.head(10))

ingest.fill_na()
ingest.change_yellow_types()
ingest.create_target()
ingest.dup_and_miss()
ingest.outlier_removal()


st.write("Distribution of target variable: Trip duration (seconds)")

fig = px.histogram(ingest.yellow_taxi_data, x="trip_duration", nbins=100, title="Trip duration distribution", labels={"trip_duration": "Trip duration (seconds)", "count": "Frequency"})

st.plotly_chart(fig, use_container_width=True)

st.write("Trip duration vs trip distance (miles) (with linear regression line")

fig1 = px.scatter(ingest.yellow_taxi_data, x="trip_distance", y="trip_duration", title="Trip duration vs trip distance", trendline="ols", trendline_color_override="red", labels={"trip_distance": "Trip distance (miles)", "trip_duration": "Trip duration (seconds)"})
st.plotly_chart(fig1, use_container_width=True)


class FeatureEngineering:

    def __init__(self, ingest):
        self.yellow_taxi_data = ingest.yellow_taxi_data
        self.green_taxi_data = ingest.green_taxi_data
        
    def one_hot(self):
        self.yellow_taxi_data = pd.concat([self.yellow_taxi_data, pd.get_dummies(self.yellow_taxi_data['store_and_fwd_flag'])], axis=1)
        self.yellow_taxi_data = pd.concat([self.yellow_taxi_data, pd.get_dummies(self.yellow_taxi_data['VendorID'])], axis=1)
        self.yellow_taxi_data.drop(['store_and_fwd_flag'], axis=1, inplace=True)
        self.yellow_taxi_data.drop(['VendorID'], axis=1, inplace=True)

    def date_features(self):
        self.yellow_taxi_data['month'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.month
        self.yellow_taxi_data['day'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.day
        self.yellow_taxi_data['hour'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.hour
        self.yellow_taxi_data['minute'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.minute
        self.yellow_taxi_data['day_of_week'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.dayofweek
        # self.yellow_taxi_data['week'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.isocalendar().week
        self.yellow_taxi_data['weekday'] = self.yellow_taxi_data.tpep_pickup_datetime.dt.weekday
        

    def drop_cols(self):
        try:
            self.yellow_taxi_data = self.yellow_taxi_data.drop(['tpep_pickup_datetime'], axis=1)
            self.yellow_taxi_data = self.yellow_taxi_data.drop(['tpep_dropoff_datetime'], axis=1)
            self.yellow_taxi_data = self.yellow_taxi_data.drop(['airport_fee'], axis=1)


        except KeyError:
            pass
        # self.yellow_taxi_data = self.yellow_taxi_data.drop(['tpep_pickup_datetime'], axis=1)
        # self.yellow_taxi_data = self.yellow_taxi_data.drop(['id'], axis=1)

        # # These cols don't exist in the kaggle dataset
        # # self.yellow_taxi_data = self.yellow_taxi_data.drop(['DOLocationID'], axis=1)
        # # self.yellow_taxi_data = self.yellow_taxi_data.drop(['PULocationID'], axis=1)
        # # self.yellow_taxi_data = self.yellow_taxi_data.drop(['airport_fee'], axis=1)
        # # self.yellow_taxi_data = self.yellow_taxi_data.drop(['RatecodeID'], axis=1)
        # # self.yellow_taxi_data = self.yellow_taxi_data.drop(['congestion_surcharge'], axis=1)
        # self.yellow_taxi_data = self.yellow_taxi_data.drop(['passenger_count'], axis=1)

    def cols_to_str(self):
        self.yellow_taxi_data.columns = self.yellow_taxi_data.columns.astype(str)
        

fe = FeatureEngineering(ingest=ingest)
fe.one_hot()
fe.date_features()

st.write(fe.yellow_taxi_data.head(10))

st.write('lol')
