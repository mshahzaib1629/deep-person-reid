import os, json
from .dataset_processors import processsors as dataset_processors


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

    return


def process_datasets(dir_path):
    """Load images present in directory by using their respective processors to extract their person ids and camera ids."""

    data_json_path = os.path.join(dir_path, "data.json")
    data = []

    if os.path.exists(data_json_path) == False:
        return []

    data_json = {}
    with open(data_json_path, "r") as json_file:
        data_json = json.load(json_file)

    for key, images in data_json.items():
        image_paths = [os.path.join(dir_path, image) for image in images]
        data.extend(dataset_processors[key](image_paths))

    return data
