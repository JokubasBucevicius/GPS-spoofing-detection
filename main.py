import os
import time
import pandas as pd
from multiprocessing import Process, Manager
import timer_wraper as tw

from data_loader import DataLoader
from task_A import LocationAnomalyDetector
from task_B import SpeedCourseAnomalyDetector
from task_C import NeighboringVesselAnomalyDetector

CHUNK_SIZE = 2_000_000
NUM_WORKERS = 4

def run_task_a(df, queue):
    detector_a = LocationAnomalyDetector(df, num_workers=NUM_WORKERS)
    jump_anomalies, invalid_jumps = detector_a.detect_location_anomalies_parallel()
    queue.put(("task_a", jump_anomalies, invalid_jumps))

def run_task_b(df, queue):
    detector_b = SpeedCourseAnomalyDetector(df, num_workers=NUM_WORKERS)
    speed_anomalies, course_anomalies = detector_b.detect_anomalies_parallel()
    queue.put(("task_b", speed_anomalies, course_anomalies))

def run_task_c(df, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies):
    detector_c = NeighboringVesselAnomalyDetector(
        df,
        jump_anomalies,
        invalid_jumps,
        speed_anomalies,
        course_anomalies,
        num_workers=NUM_WORKERS
    )
    inconsistencies = detector_c.detect_inconsistencies_parallel()
    return inconsistencies

def process_chunk(df_chunk, chunk_id, total_counts, all_inconsistencies, all_jump_anomalies, all_invalid_jumps, all_speed_anomalies, all_course_anomalies):
    print(f"\n🚀 Processing chunk {chunk_id + 1} ({len(df_chunk):,} rows)...")

    with Manager() as manager:
        queue = manager.Queue()

        # Run Task A and B in parallel
        p1 = Process(target=run_task_a, args=(df_chunk, queue))
        p2 = Process(target=run_task_b, args=(df_chunk, queue))

        p1.start()
        p2.start()
        p1.join()
        p2.join()

        results = {}
        while not queue.empty():
            task_name, anomalies_1, anomalies_2 = queue.get()
            results[task_name] = (anomalies_1, anomalies_2)

        jump_anomalies, invalid_jumps = results.get("task_a", (pd.DataFrame(), pd.DataFrame()))
        speed_anomalies, course_anomalies = results.get("task_b", (pd.DataFrame(), pd.DataFrame()))

        # Save per-task results per chunk
        jump_anomalies.to_csv(f"results/task_a_jump_anomalies_chunk{chunk_id}.csv", index=False)
        invalid_jumps.to_csv(f"results/task_a_invalid_jumps_chunk{chunk_id}.csv", index=False)
        speed_anomalies.to_csv(f"results/task_b_speed_anomalies_chunk{chunk_id}.csv", index=False)
        course_anomalies.to_csv(f"results/task_b_course_anomalies_chunk{chunk_id}.csv", index=False)

        print(" Task A & B complete.")
        print(f"  - Jump Anomalies: {len(jump_anomalies)}")
        print(f"  - Invalid Jumps: {len(invalid_jumps)}")
        print(f"  - Speed Anomalies: {len(speed_anomalies)}")
        print(f"  - Course Anomalies: {len(course_anomalies)}")

        # Task C
        print(" Running Task C...")
        inconsistencies = run_task_c(df_chunk, jump_anomalies, invalid_jumps, speed_anomalies, course_anomalies)
        print(f"  - Inconsistencies: {len(inconsistencies)}")

        # Save cumulative inconsistencies
        if not jump_anomalies.empty:
            all_jump_anomalies.append(jump_anomalies)
        if not invalid_jumps.empty:
            all_invalid_jumps.append(invalid_jumps)
        if not speed_anomalies.empty:
            all_speed_anomalies.append(speed_anomalies)
        if not course_anomalies.empty:
            all_course_anomalies.append(course_anomalies)
        if not inconsistencies.empty:
            all_inconsistencies.append(inconsistencies)

        # Update total counts
        total_counts["jump_anomalies"] += len(jump_anomalies)
        total_counts["invalid_jumps"] += len(invalid_jumps)
        total_counts["speed_anomalies"] += len(speed_anomalies)
        total_counts["course_anomalies"] += len(course_anomalies)
        total_counts["inconsistencies"] += len(inconsistencies)
@tw.timeit
def main():
    os.makedirs("results", exist_ok=True)

    print(" Loading full dataset...")
    loader = DataLoader()
    df = loader.load_data()
    total_rows = len(df)
    print(f" Full dataset loaded: {total_rows:,} rows")

    num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_counts = {
        "jump_anomalies": 0,
        "invalid_jumps": 0,
        "speed_anomalies": 0,
        "course_anomalies": 0,
        "inconsistencies": 0
    }

    all_jump_anomalies = []
    all_invalid_jumps = []
    all_speed_anomalies = []
    all_course_anomalies = []
    all_inconsistencies = []

    for i in range(num_chunks):
        start = i * CHUNK_SIZE
        end = min((i + 1) * CHUNK_SIZE, total_rows)
        df_chunk = df.iloc[start:end].copy()

        process_chunk(df_chunk, i, total_counts, all_inconsistencies,
            all_jump_anomalies, all_invalid_jumps,
            all_speed_anomalies, all_course_anomalies
        )

    # Save full inconsistencies to one file
    if all_jump_anomalies:
        pd.concat(all_jump_anomalies).to_csv("results/task_a_all_jump_anomalies.csv", index=False)
    if all_invalid_jumps:
        pd.concat(all_invalid_jumps).to_csv("results/task_a_all_invalid_jumps.csv", index=False)
    if all_speed_anomalies:
        pd.concat(all_speed_anomalies).to_csv("results/task_b_all_speed_anomalies.csv", index=False)
    if all_course_anomalies:
        pd.concat(all_course_anomalies).to_csv("results/task_b_all_course_anomalies.csv", index=False)
    if all_inconsistencies:
        pd.concat(all_inconsistencies).to_csv("results/task_c_all_inconsistencies.csv", index=False)

    # Save summary
    summary_path = "results/anomaly_summary_total.csv"
    pd.DataFrame([total_counts]).to_csv(summary_path, index=False)

    # Print summary
    print("\n TOTAL ANOMALIES DETECTED ACROSS ALL CHUNKS:")
    for k, v in total_counts.items():
        print(f"  - {k.replace('_', ' ').title()}: {v:,}")

    print(f"\n All chunks processed. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()



