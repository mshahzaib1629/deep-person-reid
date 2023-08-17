import os

def find_common_labels(folder_a, folder_b):
    # Get the list of files in folder A
    files_a = os.listdir(folder_a)
    # Get the list of files in folder B
    files_b = os.listdir(folder_b)

    # Extract unique labels from the file names in each folder
    labels_a = set(file_name[:4] for file_name in files_a)
    labels_b = set(file_name[:4] for file_name in files_b)

    # Find the common labels in both folders
    common_labels = labels_a.intersection(labels_b)

    return common_labels

# query vs test images
folder_a = "./reid-data/dukemtmc-reid/DukeMTMC-reID/query"
folder_b = "./reid-data/dukemtmc-reid/DukeMTMC-reID/bounding_box_test"

# train vs test images
# folder_a = "./reid-data/dukemtmc-reid/DukeMTMC-reID/bounding_box_train"
# folder_b = "./reid-data/dukemtmc-reid/DukeMTMC-reID/bounding_box_test"

# train vs query images
# folder_a = "./reid-data/dukemtmc-reid/DukeMTMC-reID/bounding_box_train"
# folder_b = "./reid-data/dukemtmc-reid/DukeMTMC-reID/query"

common_labels = find_common_labels(folder_a, folder_b)
print("Common labels in both folders:", common_labels)
print("Total no. of common labels: ", len(common_labels))
