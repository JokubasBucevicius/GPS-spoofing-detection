import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset (Replace 'your_vessel_data.csv' with actual file path)
df = pd.read_csv("preview.csv")  

# Ensure valid timestamps and sort by timestamp
df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
df = df.sort_values(by='# Timestamp')

# Filter out invalid coordinates
df = df[(df['Latitude'] <= 90) & (df['Longitude'] <= 180)]

# Define grid size (0.05° ~ 5.5 km x 3.5 km)
grid_size = 0.4

# Assign grid cells
df['Grid_X'] = (df['Longitude'] // grid_size).astype(int)
df['Grid_Y'] = (df['Latitude'] // grid_size).astype(int)

# Get grid boundaries
x_min, x_max = df['Longitude'].min(), df['Longitude'].max()
y_min, y_max = df['Latitude'].min(), df['Latitude'].max()

# Generate grid lines
x_ticks = np.arange(x_min, x_max, grid_size)
y_ticks = np.arange(y_min, y_max, grid_size)

# Plot vessel positions
plt.figure(figsize=(12, 8))
plt.scatter(df['Longitude'], df['Latitude'], alpha=0.5, s=10, c='blue', label="Vessels")

# Overlay grid lines
for x in x_ticks:
    plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
for y in y_ticks:
    plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)

# Labels and styling
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Vessel Positions with Medium-Size Grid (0.05°)")
plt.legend()
plt.grid(True)

# Show the map with grid overlay
plt.show()


