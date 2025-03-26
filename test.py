import time
import os
from data_loader import DataLoader
from task_A import LocationAnomalyDetector
from task_B import SpeedCourseAnomalyDetector
from task_C import NeighboringVesselAnomalyDetector
from resource_tracker import track_resources
from multiprocessing import Process
import pandas as pd

def run_pipeline_sequential(df):
    print(f"\n Running pipeline in SEQUENTIAL mode...")

    start = time.time()

    # Task A
    detector_a = LocationAnomalyDetector(df)
    jump_anomalies, invalid_jumps = detector_a.detect_location_anomalies_sequential()
    jump_anomalies.to_csv("results/task_a_jump_anomalies_seq.csv", index=False)
    invalid_jumps.to_csv("results/task_a_invalid_jumps_seq.csv", index=False)

    # Task B
    detector_b = SpeedCourseAnomalyDetector(df)
    speed_anomalies, course_anomalies = detector_b.detect_anomalies_sequential()
    speed_anomalies.to_csv("results/task_b_speed_anomalies_seq.csv", index=False)
    course_anomalies.to_csv("results/task_b_course_anomalies_seq.csv", index=False)

    # Task C
    detector_c = NeighboringVesselAnomalyDetector(
        df,
        jump_anomalies,
        invalid_jumps,
        speed_anomalies,
        course_anomalies
    )
    _ = detector_c.detect_inconsistencies_sequential()

    end = time.time()
    duration = round(end - start, 2)
    print(f"Sequential pipeline completed in {duration} seconds")
    return duration

def run_task_a(df, n_workers):
    detector_a = LocationAnomalyDetector(df, num_workers=n_workers)
    jump_anomalies, invalid_jumps = detector_a.detect_location_anomalies_parallel()
    jump_anomalies.to_csv("results/task_a_jump_anomalies.csv", index=False)
    invalid_jumps.to_csv("results/task_a_invalid_jumps.csv", index=False)

def run_task_b(df, n_workers):
    detector_b = SpeedCourseAnomalyDetector(df, num_workers=n_workers)
    speed_anomalies, course_anomalies = detector_b.detect_anomalies_parallel()
    speed_anomalies.to_csv("results/task_b_speed_anomalies.csv", index=False)
    course_anomalies.to_csv("results/task_b_course_anomalies.csv", index=False)

def run_task_c(df, n_workers):
    print(" Running Task C...")

    try:
        jump_anomalies = pd.read_csv("results/task_a_jump_anomalies.csv")
        invalid_jumps = pd.read_csv("results/task_a_invalid_jumps.csv")
        speed_anomalies = pd.read_csv("results/task_b_speed_anomalies.csv")
        course_anomalies = pd.read_csv("results/task_b_course_anomalies.csv")
    except Exception as e:
        print(f"Failed to load anomaly results: {e}")
        return

    detector_c = NeighboringVesselAnomalyDetector(
        df,
        jump_anomalies,
        invalid_jumps,
        speed_anomalies,
        course_anomalies,
        num_workers=n_workers
    )
    _ = detector_c.detect_inconsistencies_parallel()

def run_pipeline_parallel(df, n_workers):
    print(f"\nRunning pipeline in PARALLEL mode with {n_workers} workers...")

    start = time.time()

    task_a_workers = n_workers // 2
    task_b_workers = n_workers - task_a_workers
    task_c_workers = n_workers  # All for Task C after A and B

    p1 = Process(target=run_task_a, args=(df, task_a_workers))
    p2 = Process(target=run_task_b, args=(df, task_b_workers))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Check exit codes
    if p1.exitcode != 0:
        print("Task A exited with error. Skipping Task C.")
        return duration

    if p2.exitcode != 0:
        print("Task B exited with error. Skipping Task C.")
        return duration

    run_task_c(df, task_c_workers)

    end = time.time()
    duration = round(end - start, 2)
    print(f"Parallel pipeline completed in {duration} seconds")
    return duration


def main():
    worker_configs = [2, 4, 8]
    chunk_sizes = [250_000, 500_000, 1_000_000, 1_500_000, 2_000_000]

    log_file = "results/pipeline_performance_with_C_log.csv"
    os.makedirs("results", exist_ok=True)
    is_new_log = not os.path.exists(log_file)

    # Load the full dataset once to avoid redundant disk reads
    print("\nLoading full dataset once...")
    loader = DataLoader()
    df_full = loader.load_data()
    print(f"Full dataset loaded: {df_full.shape[0]:,} rows")

    with open(log_file, "a") as f:
        if is_new_log:
            f.write("n_workers,chunk_size,parallel_time,seq_time,speedup\n")

        for chunk_size in chunk_sizes:
            if chunk_size > len(df_full):
                print(f"Skipping chunk_size {chunk_size:,} — not enough rows.")
                continue

            # Slice the loaded dataset to the current chunk size
            df = df_full.iloc[:chunk_size].copy()
            print(f"\nBenchmarking with {chunk_size:,} rows")

            for n_workers in worker_configs:
                try:
                    # Dynamically wrap the pipeline with resource tracking
                    wrapped_parallel = track_resources(chunk_size=chunk_size, num_workers=n_workers)(run_pipeline_parallel)
                    par_time = wrapped_parallel(df, n_workers)

                    # Run the same pipeline in sequential mode
                    seq_time = run_pipeline_sequential(df)

                    # Compute and log speedup
                    speedup = round(seq_time / par_time, 2) if par_time > 0 else "Inf"
                    f.write(f"{n_workers},{chunk_size},{par_time},{seq_time},{speedup}\n")

                except Exception as e:
                    print(f"Benchmark failed: {e}")
                    f.write(f"{n_workers},{chunk_size},ERROR,ERROR,ERROR\n")

if __name__ == "__main__":
    main()


