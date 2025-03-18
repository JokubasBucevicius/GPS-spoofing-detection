"""Main file for loading AIS data"""

import pandas as pd
import multiprocessing as mp
from config import FILE_PATH, CHUNK_SIZE, NUM_WORKERS, DATASET_SIZE, LOAD_ALL
import timer_wraper as tw
import time

LOADING_CHUNK_SIZE = CHUNK_SIZE//NUM_WORKERS
class DataLoader:
    def __init__(self, file_path=FILE_PATH, chunk_size=CHUNK_SIZE, num_cores=NUM_WORKERS, dataset_size = DATASET_SIZE, loading_size = LOADING_CHUNK_SIZE):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.num_cores = num_cores
        self.dataset_size = dataset_size
        self.loading_size = loading_size

        # Load column names (assumes the first row contains headers)
        self.column_names = pd.read_csv(self.file_path, nrows=0).columns.tolist()

    def load_chunk(self, start_row):
        """Loads a chunk of data"""
        chunk = pd.read_csv(self.file_path, skiprows=start_row, nrows=self.chunk_size, names = self.column_names, header = None, low_memory=False)
        return chunk
    

    def load_chunk_batches(self, start_row):
        b_chunk = pd.read_csv(self.file_path, skiprows=start_row, nrows=self.loading_size, names = self.column_names, header = None, low_memory=False)
        return b_chunk
    
    # @tw.timeit
    def load_data_parallel(self):
        """
        Loads data based on `LOAD_ALL` mode:
        - If `LOAD_ALL=True`, loads the entire dataset in parallel.
        - If `LOAD_ALL=False`, loads one `CHUNK_SIZE` at a time, splitting into smaller chunks.
        """
        if LOAD_ALL:
            # Load entire dataset in parallel
            chunk_positions = list(range(1, self.dataset_size, self.chunk_size))

            with mp.Pool(processes=self.num_cores) as pool:
                results = pool.map(self.load_chunk, chunk_positions)

            df = pd.concat(results, ignore_index=True)
            print(f"Full dataset loaded: {df.memory_usage(deep=True).sum() / 1e6} MB")
            return df

        else:
            # Batch Mode: Process dataset in `CHUNK_SIZE` batches, loading `loading_size` sub-chunks at a time
            for chunk_start in range(1, self.dataset_size, self.chunk_size):  # Iterate over full dataset
                start_time = time.perf_counter()  # Start timing for batch

                sub_chunk_positions = list(range(chunk_start, chunk_start + self.chunk_size, self.chunk_size // self.num_cores))

                with mp.Pool(processes=self.num_cores) as pool:
                    results = pool.map(self.load_chunk_batches, sub_chunk_positions)

                df_chunk = pd.concat(results, ignore_index=True)
                elapsed_time = time.perf_counter() - start_time  # Stop timing after batch load

                #  Log execution time in `execution_times`
                tw.execution_times.append((f"load_batch {chunk_start}-{chunk_start + self.chunk_size}", elapsed_time))

                #  Also log execution time in file
                with open("execution_times.log", "a") as log_file:
                    log_file.write(f"Function load_batch {chunk_start}-{chunk_start + self.chunk_size} took {elapsed_time:.4f} seconds\n")

                #  Print in the same format as `@tw.timeit`
                print(f"Function load_batch {chunk_start}-{chunk_start + self.chunk_size} took {elapsed_time:.4f} seconds")

                yield df_chunk  #Yield each batch for processing

