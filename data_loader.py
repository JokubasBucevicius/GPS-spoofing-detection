import pandas as pd
from config import FILE_PATH
import timer_wraper as tw
import dask.dataframe as dd
from dask.diagnostics import ProgressBar



class DataLoader:
    def __init__(self, file_path=FILE_PATH):
        self.file_path = file_path
        
        # Load column names (assumes the first row contains headers)
        self.column_names = pd.read_csv(self.file_path, nrows=0).columns.tolist()

    @tw.timeit
    def load_data(self):
        dtypes = {
            "Timestamp": "object",  
            "Type of mobile": "category",  
            "MMSI": "int64",  
            "Latitude": "float64",  
            "Longitude": "float64",  
            "Navigational status": "category",  
            "ROT": "float64",  
            "SOG": "float64",  
            "COG": "float64",  
            "Heading": "float64",  
            "IMO": "object",  
            "Callsign": "object",  
            "Name": "object",  
            "Ship type": "category",  
            "Cargo type": "object",  
            "Width": "float64",  
            "Length": "float64",  
            "Type of position fixing device": "category",  
            "Draught": "float64",  
            "Destination": "object",  
            "ETA": "object",  
            "Data source type": "category",  
            "A": "float64",  
            "B": "float64",  
            "C": "float64",  
            "D": "float64",  
            }
        with ProgressBar():
            df = dd.read_csv(self.file_path, blocksize="75MB", dtype=dtypes, assume_missing=True, usecols = ["# Timestamp", "MMSI", "Latitude", "Longitude", "ROT", "SOG", "COG"] )
            df_cleaned = df.dropna(subset=['Latitude', 'Longitude', '# Timestamp'])
            df_cleaned = df_cleaned.drop_duplicates()
            result = df_cleaned.compute()  
        print(f"Full dataset loaded and cleaned. Total size: {result.memory_usage(deep=True).sum() / 1e6} MB")
        print(result.shape)
        return result
        

