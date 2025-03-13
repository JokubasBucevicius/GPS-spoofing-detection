"""
Main file for cleaning data (droping useless columns, droping NA values)
"""
import pandas as pd
from parallel_processor import ParallelProcessing
from config import DROP_COLUMNS, NUM_WORKERS
import timer_wraper as tw

class DataCleaner:
    def __init__(self, drop_columns=DROP_COLUMNS, num_workers=NUM_WORKERS):
        self.drop_columns = drop_columns
        self.num_workers = num_workers

    def clean_chunk(self, chunk):
        """Cleans a single chunk: drops unnecessary columns, handles missing values."""
        chunk = chunk.drop(columns=self.drop_columns, errors='ignore')

        if '# Timestamp' in chunk.columns:
            chunk['# Timestamp'] = pd.to_datetime(chunk['# Timestamp'], errors='coerce')

        chunk.dropna(subset=['Latitude', 'Longitude', '# Timestamp'], inplace=True)
        #chunk.drop_duplicates(inplace=True)

        return chunk
    @tw.timeit
    def clean_data_parallel(self, df):
        """Cleans data in parallel using multiprocessing."""
        chunk_size = len(df) // self.num_workers
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        processor = ParallelProcessing(self.num_workers)
        cleaned_chunks = processor.run_parallel(self.clean_chunk, chunks)
        processor.close()

        cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        return cleaned_df
