import pickle
import matplotlib.pyplot as plt
from PIL import Image

def display_top_n_matches(query_image_path, sorted_distances, n=5):
    # Load the sorted distances from the pickle file
    with open(sorted_distances, 'rb') as f:
        sorted_distances_list = pickle.load(f)

    # Get the top n matches
    top_n_matches = sorted_distances_list[:n]

    # Calculate the number of rows needed
    rows = (n + 4) // 5

    # Create a new figure
    fig = plt.figure(figsize=(20, 6 * rows))

    # Display the query image in the first row
    query_image = Image.open(query_image_path)
    ax = plt.subplot(rows + 1, 5, 1)
    ax.set_title("Query Image")
    plt.imshow(query_image)
    plt.axis('off')

    # Display the top n matches in the subsequent rows
    for i, match in enumerate(top_n_matches):
        match_image = Image.open(match[0])
        ax = plt.subplot(rows + 1, 5, 6 + i)
        ax.set_title(f"Match {i + 1}, Dist: {float(match[1]):.4f}")
        plt.imshow(match_image)
        plt.axis('off')

    plt.show()


# Example usage
query_image_path = './oxford/all_souls_000026.jpg'
sorted_distances_pkl = 'sorted_distances_1.pkl'
display_top_n_matches(query_image_path, sorted_distances_pkl, n=10)
