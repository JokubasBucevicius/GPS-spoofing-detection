"""
Central worker manager
"""
import multiprocessing as mp


class ParallelProcessing:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.pool = mp.Pool(processes=num_workers) # Creating worker pool

    def run_parallel(self, func, data_chunks):
        """
        Runs a function in parallel across multiple workers.
        Each worker processes a chunk of data.
        """
        results = self.pool.map(func, data_chunks)
        return results
    
    def close(self):
        """Closes the worker pool."""
        self.pool.close()
        self.pool.join()


