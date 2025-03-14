"""
Main control file to run the program
"""
from pipeline import run_all_tasks
import timer_wraper as tw

if __name__ == "__main__":
    run_all_tasks()
    tw.print_execution_summary()
