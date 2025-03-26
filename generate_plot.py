import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    "n_workers": [2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8],
    "chunk_size": [250000, 250000, 250000,
                   500000, 500000, 500000,
                   1000000, 1000000, 1000000,
                   1500000, 1500000, 1500000,
                   2000000, 2000000, 2000000],
    "parallel_time": [15.32, 8.58, 7.13,
                      13.91, 9.98, 8.4,
                      17.14, 12.73, 12.53,
                      20.17, 15.18, 16.4,
                      21.96, 16.86, 21.33],
    "seq_time": [17.94, 17.49, 17.69,
                 21.25, 20.55, 21.22,
                 22.97, 23.43, 24.09,
                 25.69, 26.02, 25.89,
                 28.43, 29.64, 28.85]
}

df = pd.DataFrame(data)

# Unique chunk sizes for x-axis
chunk_sizes = sorted(df["chunk_size"].unique())

# Plot setup
plt.figure(figsize=(10, 6))

# Plot sequential as a line (only once per chunk size)
seq_avg = df.groupby("chunk_size")["seq_time"].mean()
plt.plot(chunk_sizes, seq_avg, label="Sequential", marker='o', linewidth=2, linestyle='--')

# Plot parallel lines for each worker count
for workers in sorted(df["n_workers"].unique()):
    sub_df = df[df["n_workers"] == workers]
    plt.plot(sub_df["chunk_size"], sub_df["parallel_time"], label=f"{workers} Workers", marker='o')

# Labels and styling
plt.title("Execution Time vs Chunk Size (by Worker Count)")
plt.xlabel("Chunk Size")
plt.ylabel("Execution Time (seconds)")
plt.xticks(chunk_sizes)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Show or save
# plt.show()
plt.savefig("plots/execution_time_plot.png")
