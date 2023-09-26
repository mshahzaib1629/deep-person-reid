import os
import numpy as np
from PIL import Image
import math
from .herding_selection import herding_selection
import json
from .utils import get_name_dataset_dict, update_data_json


class AdjustNewImages:
    """
    This class is being used to add new images into the representative memory

    New classes are being added in representative memory w.r.t selection_percent out of total samples belonging to that class.
    """

    def __init__(
        self,
        current_dataset_name,
        representative__memory_main_directory,
        train_images,
        selection_percent=0.5,
        label_start_index=0,
        label_end_index=3,
    ) -> None:
        self.representative_memory_directory = os.path.join(
            representative__memory_main_directory, "memory"
        )
        self.current_dataset_name = current_dataset_name
        self.train_images = self.exclude_representative_memory_images(train_images)
        self.selection_percent = selection_percent
        self.label_start_index = label_start_index
        self.label_end_index = label_end_index + 1

    name_dataset_dict = {}

    def exclude_representative_memory_images(self, train_images):
        """train_images also contain images from representative memory (if it's used). This function will exclude them and only return new images"""
        new_images = [
            i
            for i in train_images
            if i["path"].find(self.representative_memory_directory) == -1
        ]
        return new_images

    def group_train_images(self):
        """Group images based on their labels/classes."""

        # Group items by their labels
        grouped_data = {}
        for image in self.train_images:
            label = image["name"][self.label_start_index : self.label_end_index]
            if label not in grouped_data:
                grouped_data[label] = []
            grouped_data[label].append(image)
        # Convert the grouped dictionary values to a list
        result = list(grouped_data.values())
        return result

    def update_labels_json(self, label_map):
        """Add the new selected image names along with their labels to the labels.json file."""
        # Specify the file path where you want to save the JSON file
        file_path = os.path.join(self.representative_memory_directory, "labels.json")

        label_json_data = {}

        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                label_json_data = json.load(json_file)
                json_file.close()

        # Adding new image names & their corresponding labels into existing ones
        for image_name, label in label_map.items():
            label_json_data[image_name] = label

        # Open the file in write mode and write the data to it
        with open(file_path, "w") as json_file:
            json.dump(label_json_data, json_file)
            json_file.close()
        print("=> labels.json updated with new images")

    def adjust_new_images(self):
        """Add new images to the representative memory based on the selection process"""
        if self.selection_percent > 1 or self.selection_percent < 0:
            raise Exception(
                "Invalid selection_percent provided. selection_percent must be between 0 and 1."
            )
        if (
            self.label_start_index >= self.label_end_index
            or self.label_start_index < 0
            or self.label_end_index < 0
        ):
            raise Exception("Invalid label index(s)")

        self.name_dataset_dict = get_name_dataset_dict(
            self.representative_memory_directory,
            self.current_dataset_name,
            [image["name"] for image in self.train_images],
        )

        grouped_images = self.group_train_images()
        label_map = {}

        for g_idx, image_group in enumerate(grouped_images):
            data = [img["vector"] for img in image_group]
            # Perform herding selection
            num_selected = math.ceil(self.selection_percent * len(image_group))
            selected_indices = herding_selection(data, num_selected)

            # print(f"selected_indices: {g_idx}", selected_indices)

            if not os.path.exists(self.representative_memory_directory):
                os.makedirs(self.representative_memory_directory)

            for selected_index in selected_indices:
                image_name = image_group[selected_index]["name"]
                label_map[image_name] = image_name[
                    self.label_start_index : self.label_end_index
                ]
                # Convert vector to image
                image_data = np.array(
                    image_group[selected_index]["vector"]
                )  # shape = (channels, height, width)
                image_data = image_data.transpose(
                    1, 2, 0
                )  # transposing (channels, height, width) -> (height, width, channels)
                image_data = (image_data * 255).astype(
                    np.uint8
                )  # Convert to uint8 data type
                image = Image.fromarray(image_data)

                # Save image with the corresponding name
                image_path = os.path.join(
                    self.representative_memory_directory, image_name
                )
                image.save(image_path)

            print(
                f"Selected {num_selected} images of {image_group[0]['name'][self.label_start_index : self.label_end_index]} out of {len(image_group)}"
            )

        print(
            f"=> New images added to representative memory ({self.representative_memory_directory})"
        )

        updated_memory_images_names = [
            filename
            for filename in os.listdir(self.representative_memory_directory)
            if filename.endswith(".jpg") or filename.endswith(".png")
        ]
        update_data_json(
            self.representative_memory_directory,
            updated_memory_images_names,
            self.name_dataset_dict,
        )
        print("=> data.json updated w.r.t all the images present in updated memroy")
        self.update_labels_json(label_map)
