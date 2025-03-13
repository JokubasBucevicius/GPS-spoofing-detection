"""Main file for loading AIS data"""

import pandas as pd
import multiprocessing as mp
from config import FILE_PATH, CHUNK_SIZE, NUM_WORKERS, DATASET_SIZE
import timer_wraper as tw

class DataLoader:
    def __init__(self, file_path=FILE_PATH, chunk_size=CHUNK_SIZE, num_cores=NUM_WORKERS, dataset_size = DATASET_SIZE):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.num_cores = num_cores
        self.dataset_size = dataset_size

        # Load column names (assumes the first row contains headers)
        self.column_names = pd.read_csv(self.file_path, nrows=0).columns.tolist()

    def load_chunk(self, start_row):
        """Loads a chunk of data"""
        return pd.read_csv(self.file_path, skiprows=start_row, nrows=self.chunk_size, names = self.column_names, header = None, low_memory=False)
    @tw.timeit
    def load_data_parallel(self):
        """Loads data in parallel using multiple workers"""
        chunk_positions = list(range(1, self.dataset_size, self.chunk_size))
        with mp.Pool(processes=self.num_cores) as pool:
            results = pool.map(self.load_chunk, chunk_positions)

        # Merge chunks
        df = pd.concat(results, ignore_index=True)
        print(f"Loaded dataset size: {df.memory_usage(deep=True).sum() / 1e6} MB")
        return df

