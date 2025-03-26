import pandas as pd
import multiprocessing as mp
import timer_wraper as tw
from config import GRID_SIZE

class NeighboringVesselAnomalyDetector:
    def __init__(self, df, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies, num_workers=4):
        self.df = df.copy()
        self.jump_anomalies = jump_anomalies
        self.invalid_jumps = invalid_jumps
        self.speed_anomalies = speed_anomalies
        self.course_anomalies = course_anomalies
        self.grid_size = GRID_SIZE
        self.num_workers = num_workers

    def check_anomaly_consistency(self, grid_data):
        if len(grid_data) < 6:
            return None

        grid_mmsi = set(grid_data["MMSI"])

        anomaly_mmsi_sets = []
        for anomaly_df in [self.jump_anomalies, self.invalid_jumps, self.speed_anomalies, self.course_anomalies]:
            if anomaly_df is not None and not anomaly_df.empty and "MMSI" in anomaly_df.columns:
                anomaly_mmsi_sets.append(set(anomaly_df["MMSI"]))

        vessels_with_anomalies = set.union(*anomaly_mmsi_sets) if anomaly_mmsi_sets else set()
        total_anomalous_vessels = len(vessels_with_anomalies & grid_mmsi)

        if (total_anomalous_vessels / len(grid_data)) >= 0.4:
            return grid_data

        return None

    def _process_grid_batch(self, grid_batch):
        results = []
        for group in grid_batch:
            res = self.check_anomaly_consistency(group)
            if res is not None and not res.empty:
                results.append(res)
        return results

    def _split_into_batches(self, items, n_batches):
        k, m = divmod(len(items), n_batches)
        return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_batches)]

    @tw.timeit
    def detect_inconsistencies_parallel(self):
        self.df['Grid_X'] = (self.df['Longitude'] // self.grid_size).astype(int)
        self.df['Grid_Y'] = (self.df['Latitude'] // self.grid_size).astype(int)

        grid_groups = [group for _, group in self.df.groupby(['Grid_X', 'Grid_Y'])]
        grid_batches = self._split_into_batches(grid_groups, self.num_workers)

        with mp.Pool(processes=self.num_workers) as pool:
            batch_results = pool.map(self._process_grid_batch, grid_batches)

        # Flatten results
        flat_results = [df for batch in batch_results for df in batch]
        return pd.concat(flat_results, ignore_index=True) if flat_results else pd.DataFrame()

    @tw.timeit
    def detect_inconsistencies_sequential(self):
        self.df['Grid_X'] = (self.df['Longitude'] // self.grid_size).astype(int)
        self.df['Grid_Y'] = (self.df['Latitude'] // self.grid_size).astype(int)

        grid_groups = [group for _, group in self.df.groupby(['Grid_X', 'Grid_Y'])]

        results = []
        for group in grid_groups:
            res = self.check_anomaly_consistency(group)
            if res is not None and not res.empty:
                results.append(res)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
