import os

def count_images_by_labels(folder_a, folder_b, labels):
    # Initialize dictionaries to store counts for each label in each folder
    counts_a = {label: 0 for label in labels}
    counts_b = {label: 0 for label in labels}

    # Get the list of files in folder A
    files_a = os.listdir(folder_a)
    # Get the list of files in folder B
    files_b = os.listdir(folder_b)

    # Iterate over the files in folder A and count the images for each label
    for file_name in files_a:
        for label in labels:
            if file_name.startswith(label):
                counts_a[label] += 1

    # Iterate over the files in folder B and count the images for each label
    for file_name in files_b:
        for label in labels:
            if file_name.startswith(label):
                counts_b[label] += 1

    return counts_a, counts_b

folder_a = "./reid-data/market1501-test/Market-1501-v15.09.15/query"
folder_b = "./reid-data/market1501-test/Market-1501-v15.09.15/query_sets/s1"
given_labels = ["0001", "0004"]

counts_in_folder_a, counts_in_folder_b = count_images_by_labels(folder_a, folder_b, given_labels)

for label in given_labels:
    print(f"\nLabel: {label}")
    print(f"Number of images in Folder A: {counts_in_folder_a[label]}")
    print(f"Number of images in Folder B: {counts_in_folder_b[label]}")
    print("----------------------------------------------------------------------------")
