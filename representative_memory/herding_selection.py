import numpy as np

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