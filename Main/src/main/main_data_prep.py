# Import all scripts that perform data tasks that includes:
# clean_data, create_datasets and split_process_data

import subprocess

if __name__ == "__main__":
    # Define the script's path
    clean_data = "../data_prep/clean_data.py"
    create_datasets = "../data_prep/create_datasets.py"
    split_process_data = "../data_prep/split_process_data.py"

    # Use subprocess to run the script
    subprocess.call(['python', clean_data])
    subprocess.call(['python', create_datasets])
    subprocess.call(['python', split_process_data])
