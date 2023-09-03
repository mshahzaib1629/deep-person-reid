import os
import numpy as np
from PIL import Image
from .retain_existing_memory import RetainExistingMemory
from .adjust_new_images import AdjustNewImages


def display_image(image_vector):
    # @TODO Replace with your image shape (with 3 color channels)
    image_shape = (256, 128, 3)

    # Reshape the vector back to an image
    image = image_vector.reshape(image_shape).astype(np.uint8)

    # Display the image
    img = Image.fromarray(image)
    img.show()

def apply_herding_selection(train_loader, representative_memory_directory, label_start_index=0, label_end_index=4, selection_percent=0.5, retain_percent=0.5):
    
    if not os.path.exists(representative_memory_directory):
            os.makedirs(representative_memory_directory)
    
    RetainExistingMemory(representative_memory_directory=representative_memory_directory, retain_percent=retain_percent).retain_existing_memory()
    AdjustNewImages(train_loader=train_loader, representative_memory_directory=representative_memory_directory, 
                    selection_percent=selection_percent, label_start_index=label_start_index, label_end_index=label_end_index 
                    ).adjust_new_images()
