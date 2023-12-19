import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def herding_selection(data, num_selected):
    selected_indices = []
    remaining_data = np.array(data)
    original_indices = np.arange(len(data))  # Keep track of original indices

    for _ in range(num_selected):
        # Calculate the mean of the remaining data
        mean_data = np.mean(remaining_data, axis=0)

        # Find the closest instance to the mean
        distances = [euclidean_distance(instance, mean_data) for instance in remaining_data]
        selected_idx = np.argmin(distances)

        # Add the original index of the selected data point
        selected_indices.append(original_indices[selected_idx])

        # Remove the selected instance and its index from remaining data and indices
        remaining_data = np.delete(remaining_data, selected_idx, axis=0)
        original_indices = np.delete(original_indices, selected_idx)
    
    return selected_indices
