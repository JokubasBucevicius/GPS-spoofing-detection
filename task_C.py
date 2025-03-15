import pandas as pd
import multiprocessing as mp
import timer_wraper as tw
from config import NUM_WORKERS, GRID_SIZE

class NeighboringVesselAnomalyDetector:
    def __init__(self, df, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies):
        """
        Initializes Task C detector with vessel data and anomalies from Task A and B.
        
        Parameters:
        - df: Full vessel dataset.
        - jump_anomalies, invalid_jumps: Anomalies from Task A.
        - speed_anomalies, course_anomalies: Anomalies from Task B.
        """
        self.df = df
        self.jump_anomalies = jump_anomalies
        self.invalid_jumps = invalid_jumps
        self.speed_anomalies = speed_anomalies
        self.course_anomalies = course_anomalies
        self.grid_size = GRID_SIZE  # Use grid size from config.py

    def check_anomaly_consistency(self, grid_data):
        """
        Checks if vessels inside the same grid cell have matching anomaly patterns.
        
        Parameters:
        - grid_data: DataFrame containing vessels in a single grid cell.

        Returns:
        - DataFrame of vessels with inconsistent anomaly detections.
        """
        if len(grid_data) < 6:  # Skip grids with only five vessels
            return None
        
        # Identify vessels in this grid that appear in anomaly datasets
        grid_mmsi = set(grid_data["MMSI"])
        
        # Get unique vessels that have anomalies
        vessels_with_anomalies = (
            grid_mmsi & set(self.jump_anomalies["MMSI"]) |
            grid_mmsi & set(self.invalid_jumps["MMSI"]) |
            grid_mmsi & set(self.speed_anomalies["MMSI"]) |
            grid_mmsi & set(self.course_anomalies["MMSI"])
        )

        total_anomalous_vessels = len(vessels_with_anomalies)

        # Flag the grid if at least 20% of vessels have anomalies
        if (total_anomalous_vessels / len(grid_data)) >= 0.2:
            return grid_data  # Return flagged grid
        

        return None  # No inconsistency detected

    @tw.timeit
    def detect_inconsistencies_parallel(self):
        """Runs anomaly consistency checks in parallel across grid cells."""
        
        # Assign vessels to grid cells
        self.df['Grid_X'] = (self.df['Longitude'] // self.grid_size).astype(int)
        self.df['Grid_Y'] = (self.df['Latitude'] // self.grid_size).astype(int)

        # Group vessels by grid cell
        grid_groups = [group for _, group in self.df.groupby(['Grid_X', 'Grid_Y'])]

        with mp.Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(self.check_anomaly_consistency, grid_groups)

        pool.close()
        pool.join()
        
        # Merge results
        inconsistent_vessels = pd.concat([df for df in results if df is not None and not df.empty], ignore_index=True) if any(df is not None and not df.empty for df in results) else pd.DataFrame()

        return inconsistent_vessels


    

