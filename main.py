"""
Main control file that joins the analysis together
"""
import timer_wraper as tw
from data_loader import DataLoader
from data_cleaner import DataCleaner
from task_A import LocationAnomalyDetector
from task_B import SpeedCourseAnomalyDetector

def main():
    print("Loading data in parallel...")
    loader = DataLoader()
    df = loader.load_data_parallel()
    print(f"Data Loaded! Shape: {df.shape}")

    print("Cleaning data in parallel...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data_parallel(df)
    print(f"Data Cleaned! Shape: {df_clean.shape}")

    print("Running Task A: Location Anomalies Detection...")
    anomaly_detector = LocationAnomalyDetector(df)
    jump_anomalies, invalid_jumps = anomaly_detector.detect_location_anomalies_parallel()

    print(f"\nLocation Anomalies Found: {len(jump_anomalies)}")
    if not jump_anomalies.empty:
        print(jump_anomalies.head())
    
    print(f"\nInvalid Coordinates Found: {len(invalid_jumps)}")
    if not invalid_jumps.empty:
        print(invalid_jumps.head())

    print("Running Task B: Speed and Course Anomalies Detection...")
    detector = SpeedCourseAnomalyDetector(df)
    speed_anomalies, course_anomalies = detector.detect_anomalies_parallel()

    print(f"\nSpeed Anomalies Found: {len(speed_anomalies)}")
    if not speed_anomalies.empty:
        print(speed_anomalies.head())

    print(f"\nCourse Anomalies Found: {len(course_anomalies)}")
    if not course_anomalies.empty:
        print(course_anomalies.head())



    tw.print_execution_summary()

if __name__ == "__main__":
    main()
