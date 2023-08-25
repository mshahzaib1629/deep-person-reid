import os
import shutil
from collections import defaultdict

def copy_files_with_prefix(src_folder, dest_folder, prefixes, delete_previous=False):
    file_list = os.listdir(src_folder)
    copied_labels = defaultdict(int)  # Dictionary to store label counts
    total_images_copied = 0
    already_existing_images = []

    if delete_previous:
        # Delete all files from the destination folder before copying new ones
        for file_name in os.listdir(dest_folder):
            file_path = os.path.join(dest_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    else: 
         # Calculate the label counts and other statistics for the existing files
        for file_name in os.listdir(dest_folder):
            if file_name.split('.')[1] != 'txt':
                label = file_name[:4]
                copied_labels[label] += 1
                total_images_copied += 1
                already_existing_images.append(file_name)

    for file_name in file_list:
        for prefix in prefixes:
            if file_name.startswith(prefix):
                src_path = os.path.join(src_folder, file_name)
                dest_path = os.path.join(dest_folder, file_name)
                if file_name not in already_existing_images:
                    shutil.copy(src_path, dest_path)
                    print(f"Copied {file_name} from {src_folder} to {dest_folder}")

                    label = file_name[:4]
                    copied_labels[label] += 1
                    total_images_copied += 1

    labels_path = os.path.join(dest_folder, "labels.txt")
    with open(labels_path, "w") as labels_file:
        labels_file.write('\n\n=== Labels in this chunk =====\n\n')
        for label, count in copied_labels.items():
            labels_file.write(f'"{label}", ')  # Include label count

        # adding summary
        labels_file.write('\n\n=== Summary ========\n\n')
        labels_file.write(f'Total Identities: {len(copied_labels.keys())}\n')  # Include label count 
        labels_file.write(f'Total Images: {total_images_copied}\n\n')    
        labels_file.write(f'Images per label -----\n\n')   
        for label, count in copied_labels.items():
            labels_file.write(f'"{label}": {count}\n')  # Include label count

# Example usage:
source_folder = "./reid-data/market1501-test/Market-1501-v15.09.15/bounding_box_train" 
destination_folder = "./reid-data/market1501-test/Market-1501-v15.09.15/train_chunks/c2"
prefixes_to_copy = ["0023", "0037", "0032", "0027", "0028", "0030", "0035" ]

copy_files_with_prefix(source_folder, destination_folder, prefixes_to_copy, delete_previous=False)

