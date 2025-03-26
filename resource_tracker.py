import time
import os
import psutil
import threading
import matplotlib.pyplot as plt
from functools import wraps
import config  # Import configuration settings

# Global dictionary to store resource usage per function
resource_usage = {}

def monitor_cpu_memory_usage(stop_event, usage_data, process):
    """Continuously logs CPU and memory usage while the function is running."""
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        cpu_usage = psutil.cpu_percent(interval=0.1, percpu=True)  # Get per-core CPU usage
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        
        usage_data["time"].append(elapsed_time)
        usage_data["cpu"].append(cpu_usage)
        usage_data["memory"].append(memory_usage)
        
        time.sleep(0.1)  # Sample every 100ms

def track_resources(chunk_size=None, num_workers=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import os

            process = psutil.Process()
            start_time = time.perf_counter()

            # Initialize CPU & Memory monitoring
            stop_event = threading.Event()
            usage_data = {"time": [], "cpu": [], "memory": []}

            monitor_thread = threading.Thread(
                target=monitor_cpu_memory_usage,
                args=(stop_event, usage_data, process)
            )
            monitor_thread.start()

            # Execute the function
            result = func(*args, **kwargs)

            # Stop monitoring
            stop_event.set()
            monitor_thread.join()

            # Final stats
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / (1024 * 1024)
            total_time = end_time - start_time

            resource_usage[func.__name__] = {
                "time": total_time,
                "memory": end_memory,
                "usage_data": usage_data
            }

            print(f"[Resource Tracker] {func.__name__} (CHUNK_SIZE={chunk_size}, NUM_WORKERS={num_workers}):")
            print(f"    Execution Time: {total_time:.4f} seconds")
            print(f"    Final Memory Usage: {end_memory:.2f}MB")
            print("    CPU & Memory Usage Data Captured...")

            # Save plot
            os.makedirs("plots", exist_ok=True)
            plot_cpu_memory_usage(usage_data, chunk_size, num_workers)
            plot_cpu_heatmap(usage_data, chunk_size, num_workers)

            return result

        return wrapper
    return decorator

def plot_cpu_memory_usage(usage_data, chunk_size, num_workers):
    """Plots CPU and memory usage over time for all CPU cores and memory."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot CPU usage
    num_cores = len(usage_data["cpu"][0])
    for core in range(num_cores):
        core_usage = [usage[core] for usage in usage_data["cpu"]]
        ax1.plot(usage_data["time"], core_usage, label=f"CPU Core {core}", linestyle="dotted")

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("CPU Usage (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create second axis for memory usage
    ax2 = ax1.twinx()
    ax2.plot(usage_data["time"], usage_data["memory"], label="Memory Usage (MB)", color="tab:red", linewidth=2)
    ax2.set_ylabel("Memory Usage (MB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title(f"CPU & Memory Usage Over Time\n(CHUNK_SIZE={chunk_size}, NUM_WORKERS={num_workers})")
    plt.grid(True)

   
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/CHUNK_SIZE_{chunk_size}_NUM_WORKERS_{num_workers}.png"
    plt.savefig(filename)
    plt.close()
    print(f"📊 Plot saved: {filename}")

def plot_cpu_heatmap(usage_data, chunk_size, num_workers):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    cpu_array = np.array(usage_data["cpu"])  # shape: [time_steps, num_cores]
    cpu_array = cpu_array.T  # shape: [num_cores, time_steps]

    fig, ax = plt.subplots(figsize=(10, 6))

    cax = ax.imshow(cpu_array, aspect="auto", cmap="viridis", interpolation="nearest",
                    extent=[usage_data["time"][0], usage_data["time"][-1], 0, cpu_array.shape[0]])

    ax.set_title(f"CPU Core Usage Heatmap\n(CHUNK_SIZE={chunk_size}, NUM_WORKERS={num_workers})")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("CPU Core")

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("CPU Usage (%)")

    # Save to plots/ directory
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/HEATMAP_CHUNK_SIZE_{chunk_size}_NUM_WORKERS_{num_workers}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap: {filename}")