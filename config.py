FILE_PATH = "aisdk-2024-07-06.csv"  # file path
CHUNK_SIZE = 100000  # Number of rows per chunk
NUM_WORKERS = 8 # Adjust based on your machine
DROP_COLUMNS = ["Type of mobile", "Callsign", "Ship type", "Draught", "Destination", "IMO", "Name", "Width", "Length", "Type of position fixing device", "Data source type", "A", "B", "C", "D", "ETA" ] # Column names that will be removed during data cleaning
DATASET_SIZE = 1000000 # Total size of the dataset (for small scale project only)
GRID_SIZE = 0.4 #Grid cell size for TASK C
SAVE_TO_CSV = False