import os
import json
import random

def get_existing_chunk_images_names (chunk_path):
    data_json_path = os.path.join(chunk_path, "data.json")
    json_data = {}
    with open(data_json_path, "r") as json_file:
        json_data = json.load(json_file)
        json_file.close()

    images = json_data.values()
    data = []
    for image_data in images:
        data = image_data
    return data

def get_chunk_images_names(chunks_paths = []):
    exclude_images = []

    for chunk_path in chunks_paths:
        chunk_images = get_existing_chunk_images_names(chunk_path)
        exclude_images.extend(chunk_images)
    return exclude_images

def chunks_generator(data_path, excluded_images_names=[]):
    # Get a list of all image filenames in the train directory
    image_filenames = os.listdir(data_path)

    # Extract unique labels from the image filenames

    unique_labels = list(set([filename.split('_')[0] for filename in image_filenames if filename not in excluded_images_names and filename.endswith((".jpg", ".jpeg", ".png", ".gif"))]))

    # Shuffle the unique labels randomly
    random.shuffle(unique_labels)
    print("unique labels: ", len(unique_labels))
    # Calculate the number of labels in each chunk (10% of the total)
    chunk_size = len(unique_labels) // 5

    # Initialize a dictionary to store the chunks
    label_chunks = {}

    # Split the shuffled labels into five chunks
    for i in range(5):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 4 else len(unique_labels)
        chunk_name = f'chunk{i + 1}'
        label_chunks[chunk_name] = unique_labels[start_idx:end_idx]

    # Convert the dictionary to JSON format
    json_data = json.dumps(label_chunks, indent=4)

    return json_data

existing_chunks_paths = []

excluded_images_names = get_chunk_images_names(existing_chunks_paths)
print("excluded_images: ", len(excluded_images_names))
data = chunks_generator("./reid-data/dukemtmc-reid/DukeMTMC-reID/bounding_box_train", [excluded_images_names])
print('DATA: \n', data)