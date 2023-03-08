from dataclasses import dataclass
import pandas as pd
import polars as pl

@dataclass
class IngestDataFromTlc:
    year: int = None
    month: str = None
    colour: str = None
    data: pl.DataFrame = None

    def read_taxi_data_from_tlc(self, year, month, colour) -> pl.DataFrame:
        """Downloads data from nyc taxi website and returns a pandas dataframe"""
        dataset_file = f"tripdata_{year}-{month:02}"
        dataset_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{colour}_{dataset_file}.parquet"
        self.data = pl.read_parquet(dataset_url)
        return self.data

@dataclass
class CleanData:
    data: pl.DataFrame = Ingest.data

    # Count the number of null values in each column
    def count_null_values(self):
        return self.data.null_count()

    # Drop columns with more than 50% null values
    def drop_columns_with_null_values(self):
        self.data = self.data.drop_nulls(threshold=0.5)
        return self.data
    
    



    
    








Ingest = IngestDataFromTlc()
# Choose which year, month and colour you want to download
# Possibily make this an interactive dropdown in streamlit 
Ingest.read_taxi_data_from_tlc(2019, "01", "yellow")
