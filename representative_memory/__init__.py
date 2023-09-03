import os
import numpy as np
from PIL import Image
from .retain_existing_memory import retain_existing_memory
from .adjust_new_images import adjust_new_images


def display_image(image_vector):
    # @TODO Replace with your image shape (with 3 color channels)
    image_shape = (256, 128, 3)

    # Reshape the vector back to an image
    image = image_vector.reshape(image_shape).astype(np.uint8)

    # Display the image
    img = Image.fromarray(image)
    img.show()

def apply_herding_selection(trainLoader, destination_directory, selection_percent=0.5, retain_percent=0.5):
    
    if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
    
    retain_existing_memory(destination_directory, retain_percent)
    adjust_new_images(trainLoader, destination_directory, selection_percent)
