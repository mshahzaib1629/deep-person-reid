import os
import numpy as np
from PIL import Image
import math
from .herding_selection import herding_selection
import json

def extract_images_from_loader(train_loader):
    images = []
    for b_index, batch in enumerate(train_loader):
        for i_index, vector in enumerate(batch["img"]):
            images.append(
                {
                    "name": batch["impath"][i_index].split("/")[-1],
                    "path": batch["impath"][i_index],
                    "vector": vector.numpy(),
                }
            )

    # Group items by their labels
    grouped_data = {}
    for image in images:
        # @TODO: set dynamic label identification
        label = image["name"][:4]
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(image)
    # Convert the grouped dictionary values to a list
    result = list(grouped_data.values())
    # print(f"image name: { images[20]['name']}")
    # display_image(images[20]['vector'])
    # print('len: ', len(images))

    return result

def update_labels_json(label_map, destination_directory):

    # Specify the file path where you want to save the JSON file
    file_path = os.path.join(destination_directory, 'labels.json')

    label_json_data = {}

    if (os.path.exists(file_path)):
         with open(file_path, "r") as json_file:
            label_json_data = json.load(json_file)
            json_file.close()

    # Adding new image names & their corresponding labels into existing ones
    for image_name, label in  label_map.items():
        label_json_data[image_name] = label
      
    # Open the file in write mode and write the data to it
    with open(file_path, "w") as json_file:
        json.dump(label_json_data, json_file)
        json_file.close()
    print("=> labels.json updated with new images")


def adjust_new_images(trainLoader, destination_directory, selection_percent): 
    if selection_percent > 1 or selection_percent < 0:
        raise Exception(
            "Invalid selection_percent provided. selection_percent must be between 0 and 1."
        )
    
    grouped_images = extract_images_from_loader(trainLoader)
    label_map = {}
    
    for g_idx, image_group in enumerate(grouped_images):
        data = [img["vector"] for img in image_group]
        # Perform herding selection
        num_selected = math.ceil(selection_percent * len(image_group))
        selected_indices = herding_selection(data, num_selected)

        # print(f"selected_indices: {g_idx}", selected_indices)

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        for selected_index in selected_indices:
            image_name = image_group[selected_index]["name"]
            label_map[image_name] = image_name[:4]
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
                destination_directory, image_name
            )
            image.save(image_path)
        
        print(f"Selected {num_selected} images of {image_group[0]['name'][:4]} out of {len(image_group)}")
    
    print(f'=> New images added to representative memory ({destination_directory})')
    update_labels_json(label_map, destination_directory)