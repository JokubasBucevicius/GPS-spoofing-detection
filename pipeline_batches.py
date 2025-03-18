import multiprocessing as mp
from multiprocessing import Manager
import pandas as pd
from data_loader import DataLoader
from data_cleaner import DataCleaner
from task_A import LocationAnomalyDetector
from task_B import SpeedCourseAnomalyDetector
from task_C import NeighboringVesselAnomalyDetector
from config import SAVE_TO_CSV, FILE_PATH, CHUNK_SIZE, NUM_WORKERS, DATASET_SIZE
import timer_wraper as tw
import resource_tracker as rt
import subprocess

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

def append_to_csv(df, filename):
    """Appends a DataFrame to a CSV file, creates the file if it doesn’t exist."""
    if not df.empty:
        df.to_csv(filename, mode="a", header=not pd.io.common.file_exists(filename), index=False)

def get_total_rows(file_path):
    """Returns the total number of rows in a file efficiently using wc -l (Linux/macOS)."""
    result = subprocess.run(["wc", "-l", file_path], capture_output=True, text=True)
    total_rows = int(result.stdout.split()[0]) - 1  # Subtract 1 for header
    return total_rows

def process_chunk(chunk, chunk_idx):
    """Processes a single chunk: Cleans data, runs Task A & B in parallel, then Task C."""
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data_parallel(chunk)

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

        # Run Task C after Task A & B
        detector_c = NeighboringVesselAnomalyDetector(df_clean, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies)
        inconsistent_vessels = detector_c.detect_inconsistencies_parallel()

        # Save results per chunk
        if SAVE_TO_CSV:
            append_to_csv(jump_anomalies, "jump_anomalies.csv")
            append_to_csv(invalid_jumps, "invalid_jumps.csv")
            append_to_csv(speed_anomalies, "speed_anomalies.csv")
            append_to_csv(course_anomalies, "course_anomalies.csv")
            append_to_csv(inconsistent_vessels, "inconsistent_vessels.csv")

        return jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies, inconsistent_vessels
    
@tw.timeit
@rt.track_resources
def run_all_tasks():
    """Executes batch processing in chunks."""
    total_rows = get_total_rows(FILE_PATH)
    processed_rows = 0
    print(f"📊 Total rows in dataset: {total_rows}")
    
    # Load the dataset in parallel
    loader = DataLoader(FILE_PATH, num_cores=NUM_WORKERS)

    for chunk_idx, chunk in enumerate(loader.load_data_parallel()):
        print(f"Processing chunk {chunk_idx + 1}...")
        process_chunk(chunk, chunk_idx + 1)

         # Update processed rows count
        processed_rows += len(chunk)

        # Calculate percentage completed
        progress = (processed_rows / total_rows) * 100
        print(f"✅ Progress: {progress:.2f}% ({processed_rows}/{total_rows} rows processed)\n")




