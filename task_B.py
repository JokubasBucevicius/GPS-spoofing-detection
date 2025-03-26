import pandas as pd
import multiprocessing as mp
import timer_wraper as tw

class SpeedCourseAnomalyDetector:
    def __init__(self, df, num_workers=4):
        self.df = df
        self.num_workers = num_workers

    def detect_anomalies_for_vessel(self, vessel_data):
        vessel_data = vessel_data.sort_values("# Timestamp").copy()

        vessel_data["sog_diff"] = vessel_data["SOG"].diff().abs().round(2)
        vessel_data["cog_diff"] = vessel_data["COG"].diff().apply(lambda x: round(min(abs(x), 360 - abs(x)), 2) if pd.notnull(x) else 0)
        

        max_speed_threshold = 50
        sudden_speed_jump = 5
        max_rot_threshold = 30
        max_cog_change = 180

        speed_anomalies = vessel_data[
            (vessel_data["SOG"] > max_speed_threshold) |
            (vessel_data["sog_diff"] > sudden_speed_jump)
        ]

        course_anomalies = vessel_data[
            (vessel_data["ROT"].abs() > max_rot_threshold) |
            (vessel_data["cog_diff"] > max_cog_change)
        ]

        return (
            speed_anomalies if not speed_anomalies.empty else None,
            course_anomalies if not course_anomalies.empty else None
        )

    def _process_batch(self, vessel_batch):
        speed_anomalies_all = []
        course_anomalies_all = []

        for vessel_data in vessel_batch:
            speed, course = self.detect_anomalies_for_vessel(vessel_data)
            if speed is not None:
                speed_anomalies_all.append(speed)
            if course is not None:
                course_anomalies_all.append(course)

        return (
            pd.concat(speed_anomalies_all) if speed_anomalies_all else None,
            pd.concat(course_anomalies_all) if course_anomalies_all else None
        )

    @tw.timeit
    def detect_anomalies_parallel(self):
        grouped = self.df.groupby("MMSI")
        vessel_list = [group for _, group in grouped]

        # Split into balanced batches for workers
        def chunk_list(lst, n_chunks):
            avg = len(lst) // n_chunks
            return [lst[i * avg:(i + 1) * avg] for i in range(n_chunks - 1)] + [lst[(n_chunks - 1) * avg:]]

        vessel_batches = chunk_list(vessel_list, self.num_workers)

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_batch, vessel_batches)

        speed_anomalies_list, course_anomalies_list = zip(*results)

        speed_anomalies = pd.concat([df for df in speed_anomalies_list if df is not None], ignore_index=True)
        course_anomalies = pd.concat([df for df in course_anomalies_list if df is not None], ignore_index=True)

        return speed_anomalies, course_anomalies

    @tw.timeit
    def detect_anomalies_sequential(self):
        grouped = self.df.groupby("MMSI")

        speed_anomalies_list = []
        course_anomalies_list = []

        for _, vessel_data in grouped:
            speed, course = self.detect_anomalies_for_vessel(vessel_data)
            if speed is not None:
                speed_anomalies_list.append(speed)
            if course is not None:
                course_anomalies_list.append(course)

        speed_anomalies = pd.concat(speed_anomalies_list, ignore_index=True) if speed_anomalies_list else pd.DataFrame()
        course_anomalies = pd.concat(course_anomalies_list, ignore_index=True) if course_anomalies_list else pd.DataFrame()

        return speed_anomalies, course_anomalies
