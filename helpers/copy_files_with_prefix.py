import os
import shutil
from collections import defaultdict
import json


def update_labels_txt(dest_folder, copied_labels, total_images_copied):
    labels_path = os.path.join(dest_folder, "labels.txt")
    with open(labels_path, "w") as labels_file:
        labels_file.write("\n\n=== Labels in this chunk =====\n\n")
        for label, count in copied_labels.items():
            labels_file.write(f'"{label}", ')  # Include label count

        # adding summary
        labels_file.write("\n\n=== Summary ========\n\n")
        labels_file.write(
            f"Total Identities: {len(copied_labels.keys())}\n"
        )  # Include label count
        labels_file.write(f"Total Images: {total_images_copied}\n\n")

        # @TODO: This info will be ambiguous when we will have same labelled images from different datasets
        labels_file.write(f"Images per label -----\n\n")
        for label, count in copied_labels.items():
            labels_file.write(f'"{label}": {count}\n')  # Include label count


def get_name_dataset_dict(
    data_json_base_path: str, new_dataset_name: str, new_images_names: list
) -> dict:
    """
    Get {image_name: dataset_name} dictionary. This can be used to filter images based on the selection mechanism. \
        Filtered images can be regrouped w.r.t dataset names in update_data_json.
    """
    existing_data_dict = {}
    name_dataset_dict = {}

    # Read existing data.json file
    data_json_path = os.path.join(data_json_base_path, "data.json")
    if os.path.exists(data_json_path):
        with open(data_json_path, "r") as json_file:
            existing_data_dict = json.load(json_file)

    # mapping existing images into {image_name: dataset_name} in name_dataset_dict
    for dataset, file_list in existing_data_dict.items():
        for file_name in file_list:
            name_dataset_dict[file_name] = dataset

    # adding new images into exisiting map
    for image_name in new_images_names:
        name_dataset_dict[image_name] = new_dataset_name

    return name_dataset_dict


def update_data_json(dest_folder, selected_images_names, name_dataset_dict):
    """Update data.json w.r.t all the images present in the selected_images_names"""
    # Select the entities from name_dataset_dict which are selected to be placed in the latest version of directory,
    # (by this, all the existing and new images must be available in name_dataset_dict along with their dataset names)
    filtered_name_dataset_dict = {
        key: value
        for key, value in name_dataset_dict.items()
        if key in selected_images_names
    }

    # group these entities based on their dataset_names
    data_dict = {}
    # Iterate through the original dictionary
    for key, value in filtered_name_dataset_dict.items():
        # If the value is not in the output dictionary, create a new list
        if value not in data_dict:
            data_dict[value] = []

        # Append the key to the list associated with the value
        data_dict[value].append(key)

    data_json_path = os.path.join(dest_folder, "data.json")

    # create a json file of this grouped data named data.json
    with open(data_json_path, "w") as json_file:
        json.dump(data_dict, json_file)


def copy_files_with_prefix(
    dataset_name,
    src_folder,
    dest_folder,
    labels,
    delete_previous=False,
):
    file_list = os.listdir(src_folder)
    copied_labels = defaultdict(int)  # Dictionary to store label counts
    total_images_copied = 0
    already_existing_images = []
    new_images_names = []

    if delete_previous:
        # Delete all files from the destination folder before copying new ones
        for file_name in os.listdir(dest_folder):
            file_path = os.path.join(dest_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    else:
        # Calculate the label counts and other statistics for the existing files
        for file_name in os.listdir(dest_folder):
            if file_name.split(".")[1] != "txt":
                label = file_name[:4]
                copied_labels[label] += 1
                total_images_copied += 1
                already_existing_images.append(file_name)

    for file_name in file_list:
        for prefix in labels:
            if file_name.startswith(prefix):
                src_path = os.path.join(src_folder, file_name)
                dest_path = os.path.join(dest_folder, file_name)
                if file_name not in already_existing_images:
                    shutil.copy(src_path, dest_path)
                    print(f"Copied {file_name} from {src_folder} to {dest_folder}")

                    label = file_name[:4]
                    copied_labels[label] += 1
                    total_images_copied += 1
                    new_images_names.append(file_name)

    update_labels_txt(dest_folder, copied_labels, total_images_copied)

    # taking all the images along with their dataset name to update data.json
    name_dataset_dict = get_name_dataset_dict(
        destination_folder, dataset_name, new_images_names
    )
    # taking all the image names that should be added in data.json
    image_names_in_directory = [
        filename
        for filename in os.listdir(dest_folder)
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif"))
    ]
    # updating data.json
    update_data_json(
        dest_folder,
        image_names_in_directory,
        name_dataset_dict,
    )


if __name__ == "__main__":
    # Example usage:
    dataset_name = "market1501"
    source_folder = "./reid-data/market1501/Market-1501-v15.09.15/bounding_box_train"
    destination_folder = "./reid-data/chunks/train/c4"
    labels = ["0023", "0037", "0032", "0027", "0028", "0030", "0035", "0332"]

    copy_files_with_prefix(
        dataset_name,
        source_folder,
        destination_folder,
        labels,
        delete_previous=False,
    )
