"""
Program to detect location anomalities described in TASK A
"""

import pandas as pd
import multiprocessing as mp
import timer_wraper as tw

class LocationAnomalyDetector:
    def __init__(self, df, num_workers=4):
        self.df = df
        self.num_workers = num_workers

    def detect_anomalies_for_vessel(self, vessel_data):
        """Detects unrealistic location jumps and invalid GPS jumps separately for a single vessel."""
        
        vessel_data = vessel_data.sort_values("# Timestamp").copy()  # Ensure chronological order

        # Compute differences in latitude and longitude
        vessel_data["lat_diff"] = vessel_data["Latitude"].diff().abs().round(3)
        vessel_data["lon_diff"] = vessel_data["Longitude"].diff().abs().round(3)

        # Define a threshold for unrealistic jumps
        threshold = 0.5  #0.5 degrees jump in lat/lon is suspicious
        jump_anomalies = vessel_data[(vessel_data["lat_diff"] > threshold) | (vessel_data["lon_diff"] > threshold)]

        # Track jumps to (91.0, 0.0) explicitly
        invalid_jumps = vessel_data[(vessel_data["Latitude"] == 91.0000) & (vessel_data["Longitude"] == 0.0000)]

        return (
            jump_anomalies if not jump_anomalies.empty else None,
            invalid_jumps if not invalid_jumps.empty else None
            )   

    def _process_batch(self, vessel_batch):
        """Process a batch of vessels sequentially in one worker."""
        batch_jump_anomalies = []
        batch_invalid_jumps = []

        for vessel in vessel_batch:
            jump, invalid = self.detect_anomalies_for_vessel(vessel)
            if jump is not None:
                batch_jump_anomalies.append(jump)
            if invalid is not None:
                batch_invalid_jumps.append(invalid)

        return (
            pd.concat(batch_jump_anomalies) if batch_jump_anomalies else None,
            pd.concat(batch_invalid_jumps) if batch_invalid_jumps else None
        )
    

    @tw.timeit
    def detect_location_anomalies_parallel(self):
        grouped = self.df.groupby("MMSI")
        vessel_list = [group for _, group in grouped]

        # Chunk vessels into batches based on num_workers
        def chunk_list(lst, n_chunks):
            avg = len(lst) // n_chunks
            return [lst[i * avg:(i + 1) * avg] for i in range(n_chunks - 1)] + [lst[(n_chunks - 1) * avg:]]

        vessel_batches = chunk_list(vessel_list, self.num_workers)

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_batch, vessel_batches)

        # Unpack and combine
        jump_anomalies_list, invalid_jumps_list = zip(*results)

        jump_anomalies = pd.concat([df for df in jump_anomalies_list if df is not None], ignore_index=True)
        invalid_jumps = pd.concat([df for df in invalid_jumps_list if df is not None], ignore_index=True)

        return jump_anomalies, invalid_jumps
    
    @tw.timeit
    def detect_location_anomalies_sequential(self):
        grouped = self.df.groupby("MMSI")
        jump_anomalies_all = []
        invalid_jumps_all = []

        for _, vessel_data in grouped:
            jump, invalid = self.detect_anomalies_for_vessel(vessel_data)
            if jump is not None:
                jump_anomalies_all.append(jump)
            if invalid is not None:
                invalid_jumps_all.append(invalid)

        jump_anomalies = pd.concat(jump_anomalies_all, ignore_index=True)
        invalid_jumps = pd.concat(invalid_jumps_all, ignore_index=True)
        return jump_anomalies, invalid_jumps