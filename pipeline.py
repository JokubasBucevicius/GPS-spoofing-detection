"""
Joins the analysis together and orchestrates the tasks
"""


import multiprocessing as mp
from multiprocessing import Manager
import pandas as pd
from data_loader import DataLoader
from data_cleaner import DataCleaner
from task_A import LocationAnomalyDetector
from task_B import SpeedCourseAnomalyDetector
from task_C import NeighboringVesselAnomalyDetector
from parallel_processor import ParallelProcessing
from config import FILE_PATH, NUM_WORKERS, SAVE_TO_CSV
import timer_wraper as tw
import resource_tracker as rt


def load_and_clean_data():
    """Loads and cleans data in parallel."""
    loader = DataLoader(FILE_PATH)
    df = loader.load_data_parallel()

    cleaner = DataCleaner()
    df_clean = cleaner.clean_data_parallel(df)

    return df_clean

def run_task_a(df, queue):
    """Runs Task A (Location Anomalies) and stores results in the queue."""
    detector = LocationAnomalyDetector(df)
    jump_anomalies, invalid_jumps = detector.detect_location_anomalies_parallel()

    queue.put(("task_a", jump_anomalies, invalid_jumps))
        
def run_task_b(df, queue):
    """Runs Task B (Speed & Course Anomalies) and stores results in the queue."""
    detector = SpeedCourseAnomalyDetector(df)
    speed_anomalies, course_anomalies = detector.detect_anomalies_parallel()

    queue.put(("task_b", speed_anomalies, course_anomalies))

@tw.timeit
@rt.track_resources
def run_all_tasks():
    """Executes the full pipeline: Load, Clean, Run Task A & B first, then Task C."""
    
    # Load and clean data
    df_clean = load_and_clean_data()  # This function is already decorated with @tw.timeit

    with Manager() as manager:
        queue = manager.Queue()

        # Start Task A & Task B in parallel
        process_a = mp.Process(target=run_task_a, args=(df_clean, queue))
        process_b = mp.Process(target=run_task_b, args=(df_clean, queue))

        process_a.start()
        process_b.start()

        process_a.join()
        process_b.join()

        results = {}
        while not queue.empty():
            
            task_name, anomalies_1, anomalies_2 = queue.get(timeout=5)
            results[task_name] = (anomalies_1, anomalies_2)


        # Extract results
        jump_anomalies, invalid_jumps = results.get("task_a", (pd.DataFrame(), pd.DataFrame()))
        speed_anomalies, course_anomalies = results.get("task_b", (pd.DataFrame(), pd.DataFrame()))

        print(f"Total Location Jump Anomalies: {len(jump_anomalies)}")
        print(f"Total Invalid GPS Jumps: {len(invalid_jumps)}")
        print(f"Total Speed Anomalies: {len(speed_anomalies)}")
        print(f"Total Course Anomalies: {len(course_anomalies)}")

        # Run Task C after Task A & B
        detector_c = NeighboringVesselAnomalyDetector(df_clean, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies)
        inconsistent_vessels = detector_c.detect_inconsistencies_parallel()  # This function is already decorated with @tw.timeit

        print(f"Total Neighboring Vessel Inconsistencies: {len(inconsistent_vessels)}\n")

        if SAVE_TO_CSV:
            jump_anomalies.to_csv("jump_anomalies.csv", index=False)
            invalid_jumps.to_csv("invalid_jumps.csv", index=False)
            speed_anomalies.to_csv("speed_anomalies.csv", index=False)
            course_anomalies.to_csv("course_anomalies.csv", index=False)
            inconsistent_vessels.to_csv("inconsistent_vessels.csv", index=False)

