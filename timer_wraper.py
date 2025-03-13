import time
from functools import wraps

# Global list to store execution times
execution_times = []


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Storing execution times in global list
        execution_times.append((func.__name__, total_time))

        # Log execution time to a file
        with open("execution_times.log", "a") as log_file:
            log_file.write(f"{func.__name__} took {total_time:.4f} seconds\n")

        # Print execution time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")

        return result
    
    return timeit_wrapper

def print_execution_summary():
    """Prints a summary of all function execution times."""
    print("\nExecution Time Summary:")

    total_time = 0


    for func_name, exec_time in execution_times:
        print(f"{func_name} took {exec_time:.4f} seconds")
        total_time += exec_time
    print(f"\nTotal Execution Time: {total_time:.4f} seconds")