import os
import numpy as np
from PIL import Image
import shutil
import math
from datetime import datetime
from .herding_selection import herding_selection

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

    print(f"=> Representative Memory backup created in {formatted_datetime}")
    return backup_directory

def retain_existing_memory(rp_memory_directory, retain_percent, backup_parent_directory="./reid-data/rp-memory-backups"):
    
    if retain_percent > 1 or retain_percent < 0:
        raise Exception(
            "Invalid selection_percent provided. selection_percent must be between 0 and 1."
        )
    
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

    print(f"=> Representative memory updated with previous data")