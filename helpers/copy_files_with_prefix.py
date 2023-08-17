import os
import shutil

# This function is supposed to copy images of given labels from one directory to another directory
def copy_files_with_prefix(src_folder, dest_folder, prefixes):
    # Get the list of files in the source folder
    file_list = os.listdir(src_folder)

    # Iterate over the files and copy those that start with any of the specified prefixes
    for file_name in file_list:
        for prefix in prefixes:
            if file_name.startswith(prefix):
                src_path = os.path.join(src_folder, file_name)
                dest_path = os.path.join(dest_folder, file_name)
                shutil.copy(src_path, dest_path)
                print(f"copied {file_name} from {src_folder} to {dest_folder}")

# Example usage:
source_folder = "./reid-data/market1501-test/Market-1501-v15.09.15/bounding_box_test" 
destination_folder = "./reid-data/market1501-test/Market-1501-v15.09.15/test_sets/s1"
prefixes_to_copy = ["0001", "0004", "0000"]

copy_files_with_prefix(source_folder, destination_folder, prefixes_to_copy)
