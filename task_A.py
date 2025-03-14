"""
Program to detect location anomalities described in TASK A
"""

import pandas as pd
import multiprocessing as mp
from config import NUM_WORKERS
import timer_wraper as tw

class LocationAnomalyDetector:
    def __init__(self, df):
        self.df = df

    def detect_anomalies_for_vessel(self, vessel_data):
        """Detects unrealistic location jumps and invalid GPS jumps separately for a single vessel."""
        
        vessel_data = vessel_data.sort_values("# Timestamp")  # Ensure chronological order

        # Compute differences in latitude and longitude
        vessel_data["lat_diff"] = vessel_data["Latitude"].diff().abs()
        vessel_data["lon_diff"] = vessel_data["Longitude"].diff().abs()

        # Define a threshold for unrealistic jumps
        threshold = 0.5  # Example: 0.5 degrees in lat/lon is suspicious
        jump_anomalies = vessel_data[(vessel_data["lat_diff"] > threshold) | (vessel_data["lon_diff"] > threshold)]

        # Track jumps to (91.0, 0.0) explicitly
        invalid_jumps = vessel_data[(vessel_data["Latitude"] == 91.0000) & (vessel_data["Longitude"] == 0.0000)]

        return jump_anomalies, invalid_jumps


    
    @tw.timeit
    def detect_location_anomalies_parallel(self):
        """Runs location anomaly detection in parallel for multiple vessels."""
        grouped = self.df.groupby("MMSI")

        # Prepare vessel data for parallel processing
        vessel_list = [group for _, group in grouped]

        with mp.Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(self.detect_anomalies_for_vessel, vessel_list)

        pool.close()
        pool.join()
        
        # Unpack results into separate lists
        jump_anomalies_list, invalid_jumps_list = zip(*results)

        # Merge anomalies separately
        jump_anomalies = pd.concat([df for df in jump_anomalies_list if df is not None], ignore_index=True)
        invalid_jumps = pd.concat([df for df in invalid_jumps_list if df is not None], ignore_index=True)

        return jump_anomalies, invalid_jumps