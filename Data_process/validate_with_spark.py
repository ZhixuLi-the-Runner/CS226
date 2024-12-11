import h5py
import time
from pyspark.sql import SparkSession

# Path to the trimmed file
trimmed_file = r"C:\spark\turn_on_led\150_merged_demos_random_trimmed.hdf5"

def load_demo_data(file_path):
    """
    Load all demos' timestep counts from the HDF5 file for validation.
    """
    data = []
    with h5py.File(file_path, 'r') as f:
        for demo_name in f["data"]:
            timesteps = f[f"data/{demo_name}/timestep_ids"].shape[0]
            data.append((demo_name, timesteps))
    return data

def validate_with_spark(data):
    """
    Validate the modified data using Spark.
    """
    spark = SparkSession.builder \
        .appName("ValidateSpark") \
        .getOrCreate()

    df = spark.createDataFrame(data, ["Demo_Name", "Timesteps"])
    df.show(20)  # Show the first 20 rows for debugging

    # Check if all demos have positive timestep counts
    is_valid = df.filter(df.Timesteps > 0).count() == len(data)

    if is_valid:
        print("All demos have valid timesteps.")
    else:
        print("Validation failed. Invalid timestep counts detected.")

    spark.stop()
    return is_valid

if __name__ == "__main__":
    # Load demo data from the trimmed file
    demo_data = load_demo_data(trimmed_file)
    print(f"Total demos: {len(demo_data)}")

    # Validate using Spark
    validation_passed = validate_with_spark(demo_data)
    if validation_passed:
        print("Validation passed.")
    else:
        print("Validation failed.")
