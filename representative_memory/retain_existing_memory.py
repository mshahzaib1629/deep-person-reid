import os
import numpy as np
from PIL import Image
import shutil
import math
from datetime import datetime
from .herding_selection import herding_selection
import json


class RetainExistingMemory:
    """
    This class is being used to take backup of the current representative memory and to retain the existing images in the representative memory.

    Existing classes are being retained w.r.t retain_percent out of total samples belonging to that class in representative memory.
    """

    label_json_data = {}

    def __init__(
        self,
        representative_memory_main_directory,
        retain_percent=0.5,
    ) -> None:
        self.representative_memory_directory = os.path.join(
            representative_memory_main_directory, "memory"
        )
        self.backup_parent_directory = os.path.join(
            representative_memory_main_directory, "backups"
        )
        self.retain_percent = retain_percent

        file_path = os.path.join(self.representative_memory_directory, "labels.json")

        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                self.label_json_data = json.load(json_file)
                json_file.close()

    def get_image_label_from_json(self, image_name):
        """Get label of image that belong to the current representative memory based on its name."""
        return self.label_json_data[image_name]

    def load_representative_images(self, memory_directory):
        """Load the images from the representative memory and group them based on their labels/classes."""
        image_filenames = [
            filename
            for filename in os.listdir(memory_directory)
            if filename.endswith(".jpg") or filename.endswith(".png")
        ]
        images = []

        for image_name in image_filenames:
            path = os.path.join(memory_directory, image_name)
            image = np.array(Image.open(path))
            images.append({"name": image_name, "path": path, "vector": image})

        # Group items by their labels
        grouped_data = {}
        for image in images:
            label = self.get_image_label_from_json(image["name"])
            if label not in grouped_data:
                grouped_data[label] = []
            grouped_data[label].append(image)
        # Convert the grouped dictionary values to a list
        result = list(grouped_data.values())
        return result

    def create_backup(self):
        """Create backup of current representative memory."""
        files_to_move = os.listdir(self.representative_memory_directory)

        if len(files_to_move) == 0:
            return

        if not os.path.exists(self.backup_parent_directory):
            os.makedirs(self.backup_parent_directory)

        current_datetime = datetime.now()

        # Format the date and time as a string (e.g., "2023-08-21-14-45-30")
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        # Create a directory with the formatted date and time
        backup_directory = os.path.join(
            self.backup_parent_directory, formatted_datetime
        )

        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        # Iterate through the files and move them to folderB
        for file_name in files_to_move:
            source_path = os.path.join(self.representative_memory_directory, file_name)
            destination_path = os.path.join(backup_directory, file_name)

            # Use shutil.move to move the file
            shutil.move(source_path, destination_path)

        print(f"=> Representative Memory backup created in {formatted_datetime}")
        return backup_directory

    def update_labels_json(self, label_map):
        """
        Update labels.json file w.r.t label_map.

        Ideally, it's used to create the labels.json in representative memory with the image names selected thorough the selection process.
        """
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
        print("=> labels.json updated with retained images")

    def retain_existing_memory(self):
        """Take backup of the current representative memory and retain the existing images in the representative memory."""
        if self.retain_percent > 1 or self.retain_percent < 0:
            raise Exception(
                "Invalid selection_percent provided. selection_percent must be between 0 and 1."
            )

        # create backup of current representative memory
        backup_directory = self.create_backup()
        current_memory_grouped_images = self.load_representative_images(
            backup_directory
        )
        label_map = {}

        for g_idx, image_group in enumerate(current_memory_grouped_images):
            data = [img["vector"] for img in image_group]
            # Perform herding selection
            num_selected = math.ceil(self.retain_percent * len(image_group))
            selected_indices = herding_selection(data, num_selected)

            for selected_index in selected_indices:
                image_name = image_group[selected_index]["name"]
                label_map[image_name] = self.get_image_label_from_json(image_name)

                # Convert vector to image
                image_data = np.array(
                    image_group[selected_index]["vector"]
                )  # shape = (height, width, channels)

                image = Image.fromarray(image_data)

                # Save image with the corresponding name
                image_path = os.path.join(
                    self.representative_memory_directory, image_name
                )
                image.save(image_path)

            print(
                f"Retained {num_selected} images of {self.get_image_label_from_json(image_group[0]['name'])} out of {len(image_group)}"
            )

        print(f"=> Representative memory updated with retained images")
        self.update_labels_json(label_map)
