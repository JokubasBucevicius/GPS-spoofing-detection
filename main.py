"""
Main control file that joins the analysis together
"""
import timer_wraper as tw
from data_loader import DataLoader
from data_cleaner import DataCleaner

def main():
    print("Loading data in parallel...")
    loader = DataLoader()
    df = loader.load_data_parallel()
    print(f"Data Loaded! Shape: {df.shape}")

    print("Cleaning data in parallel...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data_parallel(df)
    print(f"Data Cleaned! Shape: {df_clean.shape}")

    tw.print_execution_summary()

if __name__ == "__main__":
    main()
