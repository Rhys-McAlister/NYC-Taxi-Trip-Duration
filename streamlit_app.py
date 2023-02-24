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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass

st.set_page_config()
st.title("NYC Taxi Trip Duration Prediction")


def fetch(dataset_url: str) -> pd.DataFrame:
    """Read taxi data from web into pandas DataFrame"""

    df = pd.read_parquet(dataset_url)
    return df

def etl_web_to_gcs(year: int, month: int, colour: str) -> None:
    """The main ETL function"""
    dataset_file = f"tripdata_{year}-{month:02}"
    dataset_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{colour}_{dataset_file}.parquet"
    df = fetch(dataset_url)
    df['trip_distance'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    return df

df = etl_web_to_gcs(2021, 1, "yellow")

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

   
    def change_yellow_types2(self):
        int_cols = ['pulocationid', 'dolocationid', 'passenger_count', 'ratecodeid', 'payment_type']
        float_cols = ['fare_amount', 'total_amount', 'trip_distance', 'extra', 'mta_tax', 'tip_amount',
         'tolls_amount', 'improvement_surcharge', 'congestion_surcharge']
        
        for col in int_cols:
            self.yellow_taxi_data[col] = self.yellow_taxi_data[col].astype('int64')
        for col in float_cols:
            self.yellow_taxi_data[col] = self.yellow_taxi_data[col].astype('float64')

        
    def create_target(self):
            self.yellow_taxi_data['tpep_pickup_datetime'] = pd.to_datetime(self.yellow_taxi_data['tpep_pickup_datetime'])
            self.yellow_taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(self.yellow_taxi_data['tpep_dropoff_datetime'])

            self.yellow_taxi_data['trip_duration'] = self.yellow_taxi_data['tpep_dropoff_datetime'] - self.yellow_taxi_data['tpep_pickup_datetime']
            self.yellow_taxi_data['trip_duration'] = self.yellow_taxi_data['trip_duration'].dt.total_seconds()


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


class FeatureEngineering:

    def __init__(self, ingest):
        self.yellow_taxi_data = ingest.yellow_taxi_data
        
    def one_hot(self):
        self.yellow_taxi_data = pd.concat([self.yellow_taxi_data, pd.get_dummies(self.yellow_taxi_data['store_and_fwd_flag'])], axis=1)
        self.yellow_taxi_data = pd.concat([self.yellow_taxi_data, pd.get_dummies(self.yellow_taxi_data['vendorid'])], axis=1)
        self.yellow_taxi_data.drop(['store_and_fwd_flag'], axis=1, inplace=True)
        self.yellow_taxi_data.drop(['vendorid'], axis=1, inplace=True)

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
    
    def cols_to_str(self):
        self.yellow_taxi_data.columns = self.yellow_taxi_data.columns.astype(str)
        
@dataclass
class Model:

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    light_gbm_mse: float
    light_gbm_train_test_score: float

    def __init__(self, fe):
        self.yellow_taxi_data = fe.yellow_taxi_data

    def train_test_split(self):
        y = self.yellow_taxi_data['trip_duration']
        X = self.yellow_taxi_data.drop(['trip_duration'], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        

    def random_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor()
        rf.fit(self.X_train, self.y_train)
        y_pred = rf.predict(self.X_test)
        print(f"Random Forest RMSE: {mean_squared_error(self.y_test, y_pred, squared=False)}")

    def light_gbm(self):
        from sklearn.metrics import mean_squared_error as MSE
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        import numpy as np
        lgbm = lgb.LGBMRegressor()
        lgbm.fit(self.X_train, self.y_train)
        self.light_gbm_train_test_score = (lgbm.score(self.X_train, self.y_train), lgbm.score(self.X_test, self.y_test))
        self.light_gbm_mse = (f"MSE: {np.sqrt(MSE(self.y_test, lgbm.predict(self.X_test)))}")
        # feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_,self.X_train.columns)), columns=['Value','Feature'])
        
        
        
       

    def light_preds(self):
        import numpy as np
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        lgbm = lgb.LGBMRegressor()
        lgbm.fit(self.X_train, self.y_train)
        test_x_data = test_fe.yellow_taxi_data.drop(['airport fee'], axis = 1)
        preds = lgbm.predict(test_x_data)
        print(preds.shape)
        return preds

    def lrrr(self):
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        print(lr.score(self.X_train, self.y_train), lr.score(self.X_test, self.y_test))
        print(f"Linear Regression RMSE: {mean_squared_error(self.y_test, lr.predict(self.X_test), squared=False)}")
        plt.scatter(self.y_test, lr.predict(self.X_test))
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.show()


def main():

    # Ingest Data
    ingest = IngestData()
    ingest.read_yellow_taxi_from_api(5000)
    ingest.fill_na()
    ingest.change_yellow_types2()
    ingest.create_target()
    ingest.dup_and_miss()
    ingest.outlier_removal()


    # Feature Engineering
    fe = FeatureEngineering(ingest=ingest)
    fe.one_hot()
    fe.date_features()
    fe.drop_cols()
    fe.cols_to_str()

    # Model
    model = Model(fe)
    model.train_test_split()
    model.light_gbm()
    # st.write(f"{model.light_gbm_mse}")
    # st.write(f"{model.light_gbm_train_test_score}")

    # prose
    st.write("Here is an example of NYC yellow taxi data taken from the New York City Open Data API.")
    st.write(ingest.yellow_taxi_data.head(10))

    st.write('The goal of this model is to make reasonably accurate predictions of the duration of a taxi trip. Lets examine the distribution of our target variable, trip duration. ')
    fig = px.histogram(ingest.yellow_taxi_data, x="trip_duration", nbins=100, title="Trip duration distribution", labels={"trip_duration": "Trip duration (seconds)", "count": "Frequency"})
    st.plotly_chart(fig, use_container_width=True)
    st.write('The distribution of our target variable is highly skewed, this is understandable as most trips are short. An interesting experiment would be to analyse how the results of our model change if we alter the distribution of our target variable to be normally distributed.')

    st.write(model.yellow_taxi_data.head(10))

    # select all rows where month is 1 and count the number of rows
    st.write(model.yellow_taxi_data[model.yellow_taxi_data['month'] == 1].shape[0])
    st.write(model.yellow_taxi_data[model.yellow_taxi_data['month'] == 1])

    # Groupby the rows where month is 1
    st.write(model.yellow_taxi_data.groupby('month')['trip_distance','total_amount', 'passenger_count'].median())


if __name__ == "__main__":
    main()


    
# st.write("Here is an example of NYC yellow taxi data taken from the New York City Open Data API.")
# st.write(ingest.yellow_taxi_data.head(10))




# st.write('The goal of this model is to make reasonably accurate predictions of the duration of a taxi trip. Lets examine the distribution of our target variable, trip duration. ')


# st.write("Distribution of target variable: Trip duration (seconds)")

# fig = px.histogram(ingest.yellow_taxi_data, x="trip_duration", nbins=100, title="Trip duration distribution", labels={"trip_duration": "Trip duration (seconds)", "count": "Frequency"})


# st.plotly_chart(fig, use_container_width=True)


# st.write("Trip duration vs trip distance (miles) (with linear regression line")

# fig1 = px.scatter(ingest.yellow_taxi_data, x="trip_distance", y="trip_duration", title="Trip duration vs trip distance", trendline="ols", trendline_color_override="red", labels={"trip_distance": "Trip distance (miles)", "trip_duration": "Trip duration (seconds)"})
# st.plotly_chart(fig1, use_container_width=True)









# month_trip_variation = px.box(fe.yellow_taxi_data, x="month", y="trip_duration", points="all")
# month_trip_variation.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
# st.plotly_chart(month_trip_variation, use_container_width=True)

# month_trip_variation_v = px.violin(fe.yellow_taxi_data, x="month", y="trip_duration", box=True, points="all", hover_data=fe.yellow_taxi_data.columns)
# st.plotly_chart(month_trip_variation_v, use_container_width=True)

# daily_trip_variation = px.box(fe.yellow_taxi_data, x="day_of_week", y="trip_duration")
# daily_trip_variation.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
# st.plotly_chart(daily_trip_variation, use_container_width=True)

# st.write(fe.yellow_taxi_data.head(10))

# st.write("Grouped by month")
# st.write("Average per month")

# st.write(fe.yellow_taxi_data.groupby('month')[['month', 'trip_duration', 'total_amount', 'passenger_count']].mean())

# st.write("Count per month")

# st.write(fe.yellow_taxi_data.groupby('month')[['trip_duration']].count())

# st.write("Grouped by day of week")
# st.write("Average per day")

# st.write(fe.yellow_taxi_data.groupby('day_of_week')[['day_of_week', 'trip_duration', 'total_amount', 'passenger_count']].mean())
# st.write("Count per day")


# st.write(fe.yellow_taxi_data.groupby('weekday')[['trip_duration']].count())

# st.write("Grouped by hour of the day")
# st.write("Average per hour")

# st.write(fe.yellow_taxi_data.groupby('hour')[['hour', 'trip_duration', 'total_amount', 'passenger_count']].mean())
# st.write("Count per hour")

# st.write(fe.yellow_taxi_data.groupby('hour')[['trip_duration']].count())

# fig5 = px.scatter(fe.yellow_taxi_data, x="trip_duration", y="total_amount", color="month")
# st.plotly_chart(fig5, use_container_width=True)

# fig6 = px.scatter(fe.yellow_taxi_data, x="trip_duration", y="total_amount", color="hour")
# st.plotly_chart(fig6, use_container_width=True)

# fig7 = px.scatter(fe.yellow_taxi_data, x="trip_duration", y="total_amount", color="day_of_week")
# st.plotly_chart(fig7, use_container_width=True)


# passenger_number_boxplot = px.box(fe.yellow_taxi_data, x="passenger_count", y="trip_duration", color="passenger_count")
# st.plotly_chart(passenger_number_boxplot, use_container_width=True)


# st.write("Average trip length per passenger number")
# st.write(fe.yellow_taxi_data.groupby('passenger_count')[['trip_duration']].mean())

# st.write("Number of trips per passenger number")
# st.write(fe.yellow_taxi_data.groupby('passenger_count')[['trip_duration']].count())














