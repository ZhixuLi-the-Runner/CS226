import h5py
import numpy as np

# Input and output file paths
input_file = r"C:\spark\turn_on_led\150_merged_demos.hdf5"
output_file = r"C:\spark\turn_on_led\150_merged_demos_random_trimmed.hdf5"

def random_trim_and_save(input_file, output_file, max_random_removal=5):
    """
    Randomly deletes rows from each demo in the input HDF5 file and saves to a new file.
    """
    with h5py.File(input_file, 'r') as f:
        # Get all demo names
        demo_names = [key for key in f["data"]]

        with h5py.File(output_file, 'w') as new_f:
            # Create the top-level group for the new file
            new_data_group = new_f.create_group("data")

            for demo_name in demo_names:
                print(f"Processing {demo_name}...")

                # Access the current demo's data group
                demo_group = f[f"data/{demo_name}"]

                # Get all indices
                total_indices = np.arange(demo_group['timestep_ids'].shape[0])

                # Randomly remove rows
                num_to_remove = np.random.randint(1, max_random_removal + 1)
                remaining_indices = (
                    total_indices[num_to_remove:]
                    if len(total_indices) > num_to_remove else total_indices
                )

                # Create a new group for this demo in the output file
                filtered_demo_group = new_data_group.create_group(demo_name)

                # Copy and trim datasets
                for dataset_name in demo_group:
                    if isinstance(demo_group[dataset_name], h5py.Dataset):  # Ensure it’s a dataset
                        original_dataset = demo_group[dataset_name]
                        trimmed_data = original_dataset[remaining_indices, ...]
                        filtered_demo_group.create_dataset(dataset_name, data=trimmed_data)
                    elif isinstance(demo_group[dataset_name], h5py.Group):  # Handle nested groups
                        nested_group = filtered_demo_group.create_group(dataset_name)
                        for nested_dataset_name in demo_group[dataset_name]:
                            original_nested_dataset = demo_group[dataset_name][nested_dataset_name]
                            trimmed_data = original_nested_dataset[remaining_indices, ...]
                            nested_group.create_dataset(nested_dataset_name, data=trimmed_data)

                print(f"Finished processing {demo_name}.")

    print(f"Trimmed data saved to {output_file}")

# Run the function
random_trim_and_save(input_file, output_file)
