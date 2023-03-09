import streamlit as st
from dataclasses import dataclass
import pandas as pd
import polars as pl
import plotly.express as px

st.set_page_config()
st.title("NYC Taxi Data Analysis")


@dataclass
class IngestDataFromTlc:
    year: int = None
    month: str = None
    colour: str = None
    data: pd.DataFrame = None


    def read_taxi_data_from_tlc(self, year, month, colour) -> pd.DataFrame:
        """Downloads data from nyc taxi website and returns a pandas dataframe"""
        dataset_file = f"tripdata_{year}-{month}"
        dataset_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{colour}_{dataset_file}.parquet"
        self.data = pd.read_parquet(dataset_url)
        return self.data
    

    # Create target varuable
    def create_target_variable(self):
        self.data["trip_duration"] = self.data["tpep_dropoff_datetime"] - self.data["tpep_pickup_datetime"]
        return self.data
    




st.write("Select the year and month you would like to examine (this may take a few seconds to load).")

year_selection = st.radio(
    "Select a year", ("2019", "2020", "2021")
)

month_selection = st.radio(
    "Select a month", ("01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12")
)

st.write(f"Year: {year_selection}")
st.write(f"Month: {month_selection}")

Ingest = IngestDataFromTlc()
# Choose which year, month and colour you want to download
# Possibily make this an interactive dropdown in streamlit 

Ingest.read_taxi_data_from_tlc(year_selection, month_selection, "yellow")


Ingest.create_target_variable()


st.write('Raw taxi data downloaded from NYC TLC website:')
st.write(Ingest.data.head(5))

# class EDAViz:

#     def __init__(self, data = Ingest.data):
#         self.data = data

#     def plot_trip_duration(self):
#         fig = px.histogram(self.data, x="trip_duration")
#         st.plotly_chart(fig)

# EDAViz = EDAViz()
# EDAViz.plot_trip_duration()

# class CleanData:
    
#     def __init__(self, data = Ingest.data):
#         self.data = data


#     # # Count the number of null values in each column
#     # def count_null_values(self):
#     #     return self.data.null_count()
    
# cleandata = CleanData()
# st.write(cleandata.data)


# class FeatureEngineering:
#     data: pl.DataFrame = Clean.data

#    # Create target variable
#     def create_target_variable(self):
#         self.data.with_columns((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).alias("trip_duration"))
    



    



