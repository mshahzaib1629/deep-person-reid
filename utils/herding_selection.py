import os
import numpy as np
from PIL import Image
import shutil
import math
from datetime import datetime

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


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
    image = image_vector.reshape(image_shape).astype(np.uint8)

    # Display the image
    img = Image.fromarray(image)
    img.show()


def extract_images_from_loader(train_loader):
    images = []
    print('in function...', train_loader)
    for b_index, batch in enumerate(train_loader):
        for i_index, vector in enumerate(batch["img"]):
            print(batch["impath"][i_index].split("/")[-1])
            # print(f'batch: {b_index} -> image: {i_index} - {batch["impath"][i_index].split("/")[-1]} - {batch["impath"][i_index]}')
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

def load_representative_images(memory_directory):
    image_filenames = [filename for filename in os.listdir(memory_directory) if filename.endswith('.jpg')]
    images = []
    
    for image_name in image_filenames:
        path = os.path.join(memory_directory, image_name)
        image = np.array(Image.open(path))
        images.append({
            "name": image_name,
            "path": path,
            "vector": image
        })

    # Group items by their labels
    grouped_data = {}
    for image in images:
        # @TODO: set dynamic label identification
        # maintain a txt file containing all the label patterns
        # these label patterns can be from various distributions
        # on loading images from representative memory, group them on the bases of their label patterns
        label = image["name"][:4]
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(image)
    # Convert the grouped dictionary values to a list
    result = list(grouped_data.values())
    # print(f"image name: { result[1][0]['name']}")
    # display_image(result[1][0]['vector'])
    return result

def create_backup(rp_memory_directory, backup_parent_directory):
   
    files_to_move = os.listdir(rp_memory_directory)
    
    if len(files_to_move) == 0:
        return

    if not os.path.exists(backup_parent_directory):
            os.makedirs(backup_parent_directory)

    current_datetime = datetime.now()

    # Format the date and time as a string (e.g., "2023-08-21-14-45-30")
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    # Create a directory with the formatted date and time
    backup_directory = os.path.join(backup_parent_directory, formatted_datetime)
    
    if not os.path.exists(backup_directory): 
        os.makedirs(backup_directory)
    
    # Iterate through the files and move them to folderB
    for file_name in files_to_move:
        source_path = os.path.join(rp_memory_directory, file_name)
        destination_path = os.path.join(backup_directory, file_name)
        
        # Use shutil.move to move the file
        shutil.move(source_path, destination_path)

    print(f"Representative Memory backup created in {formatted_datetime}")
    return backup_directory

def retain_existing_memory(rp_memory_directory, retain_percent, backup_parent_directory="./reid-data/rp-memory-backups"):
    
    # create backup of current representative memory
    backup_directory = create_backup(rp_memory_directory, backup_parent_directory)

    current_memory_grouped_images = load_representative_images(backup_directory)
    

    for g_idx, image_group in enumerate(current_memory_grouped_images):
        data = [img["vector"] for img in image_group]
        # Perform herding selection
        num_selected = math.ceil(retain_percent * len(image_group))
        selected_indices = herding_selection(data, num_selected)

        for selected_index in selected_indices:
            # Convert vector to image
            image_data = np.array(
                image_group[selected_index]["vector"]
            )  # shape = (height, width, channels)
        
            image = Image.fromarray(image_data)

            # Save image with the corresponding name
            image_path = os.path.join(
                rp_memory_directory, image_group[selected_index]["name"]
            )
            image.save(image_path)

    
    print(
            f"Representative memory updated with previous data"
        )
    return

def apply_herding_selection(trainLoader, destination_directory, selection_percent=0.5, retain_percent=0.5):
    
    if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

    
    if selection_percent > 1 or selection_percent < 0:
        raise Exception(
            "Invalid selection_percent provided. selection_percent must be between 0 and 1."
        )
    
    retain_existing_memory(destination_directory, retain_percent)

    grouped_images = extract_images_from_loader(trainLoader)

    for g_idx, image_group in enumerate(grouped_images):
        data = [img["vector"] for img in image_group]
        # Perform herding selection
        num_selected = math.ceil(selection_percent * len(image_group))
        selected_indices = herding_selection(data, num_selected)

        # print(f"selected_indices: {g_idx}", selected_indices)

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        for selected_index in selected_indices:
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
                destination_directory, image_group[selected_index]["name"]
            )
            image.save(image_path)

        print(
            f"Selected {num_selected} images of {image_group[0]['name'][:4]} and saved to {destination_directory}"
        )
