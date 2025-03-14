import pandas as pd
import numpy as np
import multiprocessing as mp
import timer_wraper as tw
from config import NUM_WORKERS



class SpeedCourseAnomalyDetector:
    def __init__(self, df):
        self.df = df
    
    def detect_anomalies_for_vessel(self, vessel_data):
        """Detects both speed and course anomalies for a single vessel."""
        
        vessel_data = vessel_data.sort_values("# Timestamp").copy()

        # Compute Speed Difference (SOG change)
        vessel_data["sog_diff"] = vessel_data["SOG"].diff().abs()

        # Compute Course Over Ground (COG) change
        vessel_data["cog_diff"] = vessel_data["COG"].diff().abs()

        # Define anomaly thresholds
        max_speed_threshold = 50  # Max realistic speed in knots
        sudden_speed_jump = 5     # Sudden speed jump threshold in knots
        max_rot_threshold = 30     # Example threshold for ROT anomaly
        max_cog_change = 100       # Large course deviation threshold

        # Identify Speed Anomalies
        speed_anomalies = vessel_data[(vessel_data["SOG"] > max_speed_threshold) | 
                                      (vessel_data["sog_diff"] > sudden_speed_jump)]

        # Identify Course Anomalies
        course_anomalies = vessel_data[(vessel_data["ROT"].abs() > max_rot_threshold) | 
                                       (vessel_data["cog_diff"] > max_cog_change)]

        return speed_anomalies if not speed_anomalies.empty else None, course_anomalies if not course_anomalies.empty else None

    @tw.timeit
    def detect_anomalies_parallel(self):
        """Runs anomaly detection in parallel for multiple vessels."""
        grouped = self.df.groupby("MMSI")

        # Prepare vessel data for parallel processing
        vessel_list = [group for _, group in grouped]

        with mp.Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(self.detect_anomalies_for_vessel, vessel_list)

        pool.close()
        pool.join()
        
        # Unpack results into separate lists
        speed_anomalies_list, course_anomalies_list = zip(*results)

        # Merge anomalies separately
        speed_anomalies = pd.concat([df for df in speed_anomalies_list if df is not None and not df.empty], ignore_index=True) if any(df is not None and not df.empty for df in speed_anomalies_list) else pd.DataFrame()
        course_anomalies = pd.concat([df for df in course_anomalies_list if df is not None and not df.empty], ignore_index=True) if any(df is not None and not df.empty for df in course_anomalies_list) else pd.DataFrame()

        return speed_anomalies, course_anomalies