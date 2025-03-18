"""
Main control file to run the program
"""
from config import LOAD_ALL
if LOAD_ALL:
    from pipeline import run_all_tasks  # Full dataset processing
else:
    from pipeline_batches import run_all_tasks  # Batch processing
import timer_wraper as tw

if __name__ == "__main__":
    run_all_tasks()
    tw.print_execution_summary()
