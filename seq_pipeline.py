"""
Sequential pipeline for vessel anomaly detection (No parallel processing).
"""

import pandas as pd
from data_loader import DataLoader
from data_cleaner import DataCleaner
from task_A import LocationAnomalyDetector
from task_B import SpeedCourseAnomalyDetector
from task_C import NeighboringVesselAnomalyDetector
from config import FILE_PATH, SAVE_TO_CSV
import timer_wraper as tw



def load_and_clean_data():
    """Loads and cleans data sequentially."""
    loader = DataLoader(FILE_PATH)
    df = loader.load_data_parallel()  # Loading sequentially
    
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data_parallel(df)

    return df_clean


@tw.timeit
def run_all_tasks():
    """Executes the full sequential pipeline."""
    
    df_clean = load_and_clean_data()

    # Task A
    detector_a = LocationAnomalyDetector(df_clean)
    jump_anomalies, invalid_jumps = detector_a.detect_location_anomalies_parallel()

    print(f"Location Jump Anomalies: {len(jump_anomalies)}")
    print(f"Invalid GPS Jumps: {len(invalid_jumps)}")

    # Task B
    detector_b = SpeedCourseAnomalyDetector(df_clean)
    speed_anomalies, course_anomalies = detector_b.detect_anomalies_parallel()

    print(f"Speed Anomalies: {len(speed_anomalies)}")
    print(f"Course Anomalies: {len(course_anomalies)}")

    # Task C
    detector_c = NeighboringVesselAnomalyDetector(df_clean, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies)
    inconsistent_vessels = detector_c.detect_inconsistencies_parallel()

    print(f"Neighboring Vessel Inconsistencies: {len(inconsistent_vessels)}")

    # Save results if enabled
    if SAVE_TO_CSV:
        jump_anomalies.to_csv("jump_anomalies.csv", index=False)
        invalid_jumps.to_csv("invalid_jumps.csv", index=False)
        speed_anomalies.to_csv("speed_anomalies.csv", index=False)
        course_anomalies.to_csv("course_anomalies.csv", index=False)
        inconsistent_vessels.to_csv("inconsistent_vessels.csv", index=False)



