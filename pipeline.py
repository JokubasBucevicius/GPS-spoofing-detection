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
from config import FILE_PATH, NUM_WORKERS
import timer_wraper as tw


def load_and_clean_data():
    """Loads and cleans data in parallel."""
    print("Loading data in parallel...")
    loader = DataLoader(FILE_PATH)
    df = loader.load_data_parallel()
    print(f"Data Loaded: {df.shape}")

    print("Cleaning data in parallel...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data_parallel(df)
    print(f"Data Cleaned: {df_clean.shape}")

    return df_clean

def run_task_a(df, queue):
    """Runs Task A (Location Anomalies) and stores results in the queue."""
    print("[Task A] Starting anomaly detection...")

    try:
        detector = LocationAnomalyDetector(df)
        jump_anomalies, invalid_jumps = detector.detect_location_anomalies_parallel()

        print(f"[Task A] Detected {len(jump_anomalies)} location jump anomalies.")
        print(f"[Task A] Detected {len(invalid_jumps)} invalid GPS jumps.")

        queue.put(("task_a", jump_anomalies, invalid_jumps))
        print(" [Task A] Results added to queue.")
        
    except Exception as e:
        print(f"[Task A] ERROR: {e}")
    
    finally:
        print("[Task A] Finished.")



def run_task_b(df, queue):
    """Runs Task B (Speed & Course Anomalies) and stores results in the queue."""
    print("[Task B] Starting anomaly detection...")

    try:
        detector = SpeedCourseAnomalyDetector(df)
        speed_anomalies, course_anomalies = detector.detect_anomalies_parallel()

        print(f"[Task B] Detected {len(speed_anomalies)} speed anomalies.")
        print(f"[Task B] Detected {len(course_anomalies)} course anomalies.")

        queue.put(("task_b", speed_anomalies, course_anomalies))
        print("[Task B] Results added to queue.")
        
    except Exception as e:
        print(f"[Task B] ERROR: {e}")
    
    finally:
        print("[Task B] Finished.")

@tw.timeit
def run_all_tasks():
    """Executes the full pipeline: Load, Clean, Run Task A & B first, then Task C."""
    
    # Load and clean data
    print("Starting data loading and cleaning...")
    df_clean = load_and_clean_data()  # This function is already decorated with @tw.timeit

    with Manager() as manager:
        queue = manager.Queue()

        # Start Task A & Task B in parallel
        print("Starting anomaly detection tasks (Task A & B)...")

        process_a = mp.Process(target=run_task_a, args=(df_clean, queue))
        process_b = mp.Process(target=run_task_b, args=(df_clean, queue))

        process_a.start()
        process_b.start()

        print("Tasks A & B started. Waiting for completion.")

        # Wait for Task A & B to complete
        process_a.join(timeout=15)
        if process_a.is_alive():
            print("Task A did not terminate in time. Forcing termination.")
            process_a.terminate()

        process_b.join(timeout=15)
        if process_b.is_alive():
            print("Task B did not terminate in time. Forcing termination.")
            process_b.terminate()

        print("Tasks A & B completed.")

        # Retrieve results from Task A & B
        queue_size = queue.qsize()
        print(f"Queue contains {queue_size} items.")

        results = {}
        while not queue.empty():
            try:
                task_name, anomalies_1, anomalies_2 = queue.get(timeout=5)
                results[task_name] = (anomalies_1, anomalies_2)
                print(f"Retrieved results for {task_name}: {len(anomalies_1)} & {len(anomalies_2)} anomalies.")

            except Exception as e:
                print(f"Error retrieving from queue: {e}")

        # Extract results
        jump_anomalies, invalid_jumps = results.get("task_a", (pd.DataFrame(), pd.DataFrame()))
        speed_anomalies, course_anomalies = results.get("task_b", (pd.DataFrame(), pd.DataFrame()))

        print(f"Total Location Jump Anomalies: {len(jump_anomalies)}")
        print(f"Total Invalid GPS Jumps: {len(invalid_jumps)}")
        print(f"Total Speed Anomalies: {len(speed_anomalies)}")
        print(f"Total Course Anomalies: {len(course_anomalies)}")

        # Run Task C after Task A & B
        print("Starting Task C: Checking anomaly consistency in neighboring vessels.")
        detector_c = NeighboringVesselAnomalyDetector(df_clean, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies)
        inconsistent_vessels = detector_c.detect_inconsistencies_parallel()  # This function is already decorated with @tw.timeit

        print(f"Total Neighboring Vessel Inconsistencies: {len(inconsistent_vessels)}")

        # Save results (Uncomment if needed)
        # jump_anomalies.to_csv("jump_anomalies.csv", index=False)
        # invalid_jumps.to_csv("invalid_jumps.csv", index=False)
        # speed_anomalies.to_csv("speed_anomalies.csv", index=False)
        # course_anomalies.to_csv("course_anomalies.csv", index=False)
        # inconsistent_vessels.to_csv("inconsistent_vessels.csv", index=False)

        print("Pipeline execution completed successfully.")

