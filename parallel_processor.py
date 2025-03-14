"""
Central worker manager
"""
import multiprocessing as mp

class ParallelProcessing:
    def __init__(self, num_workers):
        """Initialize the multiprocessing worker pool."""
        self.num_workers = num_workers
        self.pool = mp.Pool(processes=num_workers)  # Creating worker pool

    def run_parallel(self, func, data_chunks):
        """Runs a function in parallel across multiple workers."""
        results = self.pool.map(func, data_chunks)
        return results

    def close(self):
        """Closes the worker pool."""
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        """Context manager entry: returns self."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures cleanup of the pool when exiting the context."""
        self.close()

    @staticmethod
    def allocate_workers(total_workers, num_tasks=2):
        """
        Allocates workers dynamically between multiple tasks.
        Example: If total_workers = 8 and num_tasks = 2, assigns 4 workers per task.
        """
        return max(1, total_workers // num_tasks)



