import os
import numpy as np
from PIL import Image
import shutil
import math


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# @TODO: Need to run this code for every label indvidually


def herding_selection(data, num_selected):
    selected_indices = []
    selected_data = []

    # Step 1: Select an initial instance (e.g., the first instance)
    selected_indices.append(0)
    selected_data.append(data[0])

    for _ in range(1, num_selected):
        # Initialize the maximum similarity and selected index
        max_similarity = -float("inf")
        selected_idx = None

        for idx, instance in enumerate(data):
            if idx in selected_indices:
                continue

            # Calculate similarity to instances in the selected subset
            similarities = [
                euclidean_distance(instance, selected) for selected in selected_data
            ]
            similarity = sum(similarities)

            # Update the selected instance if the similarity is greater
            if similarity > max_similarity:
                max_similarity = similarity
                selected_idx = idx

        # Add the selected instance to the subset
        selected_indices.append(selected_idx)
        selected_data.append(data[selected_idx])

    return selected_indices


def display_image(image_vector):
    # @TODO Replace with your image shape (with 3 color channels)
    image_shape = (256, 128, 3)

    # Reshape the vector back to an image
    array = image_vector.numpy()
    image = array.reshape(image_shape).astype(np.uint8)

    # Display the image
    img = Image.fromarray(image)
    img.show()


def extract_images_from_loader(train_loader):
    images = []
    for b_index, batch in enumerate(train_loader):
        for i_index, vector in enumerate(batch["img"]):
            # print(f'batch: {b_index} -> image: {i_index} - {batch["impath"][i_index].split("/")[-1]} - {batch["impath"][i_index]}')
            images.append(
                {
                    "name": batch["impath"][i_index].split("/")[-1],
                    "path": batch["impath"][i_index],
                    "vector": vector,
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


def apply_herding_selection(trainLoader, destination_directory, selection_percent=0.5):
    if selection_percent > 1 or selection_percent < 0:
        raise Exception(
            "Invalid selection_percent provided. selection_percent must be between 0 and 1."
        )
    grouped_images = extract_images_from_loader(trainLoader)

    for g_idx, image_group in enumerate(grouped_images):
        data = [img["vector"].numpy() for img in image_group]
        # Perform herding selection
        num_selected = math.ceil(selection_percent * len(image_group))
        selected_indices = herding_selection(data, num_selected)

        # print(f"selected_indices: {g_idx}", selected_indices)

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        for selected_index in selected_indices:
            # Convert vector to image
            image_data = np.array(
                image_group[selected_index]["vector"].numpy()
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
                destination_directory, image_group[selected_index]["name"]
            )
            image.save(image_path)

        print(
            f"Selected {num_selected} images of {image_group[0]['name'][:4]} and saved to {destination_directory}"
        )
