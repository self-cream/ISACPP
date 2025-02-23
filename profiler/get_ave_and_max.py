import os
import pandas as pd

def find_ave_and_max_values_in_csv_files(path):
    # Initialize an empty dictionary to store max values for each column
    max_values = {}

    # Iterate through all files in the directory and subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                # Construct the full path to the CSV file
                csv_file_path = os.path.join(root, file)

                # Read the CSV file into a Pandas DataFrame
                df = pd.read_csv(csv_file_path)

                # Calculate the maximum value for each column in the DataFrame
                file_max_values = df.max()
                file_avg_values = df.mean()

                print("------------------------------------------")
                print("csv file path: %s" %csv_file_path)
                print("maximum value in this file:")
                print(file_max_values)
                print("average value in this file:")
                print(file_avg_values)


# Specify the directory path to start the search
gpu_mem_directory_path = '/home/lzj/test-dcgm/gpu-monitor-data/gpu-mem'
gpu_util_directory_path = '/home/lzj/test-dcgm/gpu-monitor-data/gpu-util'
pcie_band_directory_path = '/home/lzj/test-dcgm/gpu-monitor-data/pcie-band'
cpu_util_directory_path = '/home/lzj/test-dcgm/gpu-monitor-data/cpu-util'


# Call the function to find maximum values in all CSV files
find_ave_and_max_values_in_csv_files(gpu_mem_directory_path)
find_ave_and_max_values_in_csv_files(gpu_util_directory_path)
find_ave_and_max_values_in_csv_files(pcie_band_directory_path)
find_ave_and_max_values_in_csv_files(cpu_util_directory_path)

