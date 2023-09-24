import os
import numpy as np
from PIL import Image
import json
from .retain_existing_memory import RetainExistingMemory
from .adjust_new_images import AdjustNewImages
from .memory_loader import RepresentativeMemory
from .chunk_loader import ChunkLoader


def display_image(image_vector):
    # @TODO Replace with your image shape (with 3 color channels)
    image_shape = (256, 128, 3)

    # Reshape the vector back to an image
    image = image_vector.reshape(image_shape).astype(np.uint8)

    # Display the image
    img = Image.fromarray(image)
    img.show()


def extract_images_from_loader(train_loader):
    """Extract images from the loader"""
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
    return images


def update_representative_memory(
    train_loader,
    representative_memory_main_directory,
    label_start_index=0,
    label_end_index=4,
    selection_percent=0.5,
    retain_percent=0.5,
):
    """
    Apply Herding Selection on new and existing images
    """
    if not os.path.exists(representative_memory_main_directory):
        os.makedirs(representative_memory_main_directory)

    train_images = extract_images_from_loader(train_loader)
    RetainExistingMemory(
        representative_memory_main_directory,
        retain_percent=retain_percent,
    ).retain_existing_memory()

    AdjustNewImages(
        representative_memory_main_directory,
        train_images=train_images,
        selection_percent=selection_percent,
        label_start_index=label_start_index,
        label_end_index=label_end_index,
    ).adjust_new_images()


def get_representative_images(representative_memory_main_directory):
    """Returns image paths and their corresponding labels."""
    image_paths = []
    labels = []
    representative_memory_directory = os.path.join(
        representative_memory_main_directory, "memory"
    )
    labels_file_path = os.path.join(representative_memory_directory, "labels.json")

    if os.path.exists(labels_file_path):
        with open(labels_file_path, "r") as json_file:
            label_json_data = json.load(json_file)
            json_file.close()

        for filename in os.listdir(representative_memory_directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(representative_memory_directory, filename)
                image_paths.append(image_path)
                labels.append(label_json_data[filename])

    return image_paths, labels
