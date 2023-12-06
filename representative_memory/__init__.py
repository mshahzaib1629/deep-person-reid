import os
import numpy as np
from PIL import Image
import json
from .retain_existing_memory import RetainExistingMemory
from .adjust_new_images import AdjustNewImages
from .memory_loader import RepresentativeMemory
from .chunk_loader import ChunkLoader


def display_image(image_vector):
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
            img = Image.open(batch["impath"][i_index])
            img = img.resize((128, 256))
            img_array = np.array(img)
            # Ensure the image has 3 channels (e.g., RGB). If not, convert it.
            if len(img_array.shape) == 2:
                # Convert grayscale to RGB
                img_array = np.stack((img_array,) * 3, axis=-1)
  
            address = os.path.normpath(batch["impath"][i_index])
            image_name = os.path.basename(address)

            images.append(
                {
                    "name": image_name,
                    "path": address,
                    "vector": img_array,
                }
            )
    return images


def seperate_new_and_rp_images(train_loader, representative_memory_directory):
    """train_images also contain images from representative memory (if it's used). This function will seperate the new and representative memory images."""

    train_images = extract_images_from_loader(train_loader)
    new_images = []
    rp_images = []
    for i in train_images:
        if (
            isinstance(representative_memory_directory, str)
            and i["path"].find(representative_memory_directory) > -1
        ):
            rp_images.append(i)
        else:
            new_images.append(i)

    return new_images, rp_images


def update_representative_memory(
    train_loader,
    current_dataset_name,
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
    if not os.path.exists(os.path.join(representative_memory_main_directory, "memory")):
        os.makedirs(os.path.join(representative_memory_main_directory, "memory"))

    new_images, rp_images = seperate_new_and_rp_images(
        train_loader, representative_memory_main_directory
    )
    RetainExistingMemory(
        representative_memory_main_directory,
        retain_percent=retain_percent,
    ).retain_existing_memory()

    AdjustNewImages(
        current_dataset_name,
        representative_memory_main_directory,
        train_images=new_images,
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
