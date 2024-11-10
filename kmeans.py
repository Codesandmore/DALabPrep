import csv
import random
import matplotlib.pyplot as plt

# Load data from a CSV file
file_path = 'kmeans.csv'  # Replace with the path to your CSV file

# Reading data from CSV
data = []
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip header row
    for row in reader:
        # Extract all attributes except the first one (assuming first column is non-feature)
        data.append([float(value) for value in row[1:]])

# Set the number of clusters (k)
k = 2

# Randomly initialize centroids from the dataset points
centroids = random.sample(data, k)

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))) ** 0.5

# Assign each point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for i in range(k)]
    for point in data:
        min_distance = float('inf')  # Use float('inf') to represent infinity
        closest_centroid_index = 0
        for i in range(k):
            distance = calculate_distance(point, centroids[i])
            if distance < min_distance:
                min_distance = distance
                closest_centroid_index = i
        clusters[closest_centroid_index].append(point)
    return clusters

# Update centroids to be the mean of assigned points
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue  # Skip empty clusters
        new_centroid = [0] * len(cluster[0])  # Create a centroid with the same number of dimensions
        for point in cluster:
            for i in range(len(point)):
                new_centroid[i] += point[i]
        for i in range(len(new_centroid)):
            new_centroid[i] /= len(cluster)
        new_centroids.append(new_centroid)
    return new_centroids

# Run the K-means clustering
def kmeans(data, k, max_iterations=100):
    centroids = random.sample(data, k)
    iterations = 0
    while iterations < max_iterations:
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:  # Stop if centroids converge
            break
        centroids = new_centroids
        iterations += 1
    return clusters, centroids

# Perform K-means clustering
clusters, centroids = kmeans(data, k)

# Print the final clusters and centroids
for i in range(len(clusters)):
    print(f"\nCluster {i + 1}:")
    for point in clusters[i]:
        print(point)
print("\nFinal centroids:", centroids)

# Plot the clusters and centroids
def plot_clusters(clusters, centroids):
    # Plotting clusters
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Add more colors if k > 6
    for i in range(len(clusters)):
        cluster = clusters[i]
        x_vals = [point[0] for point in cluster]  # First attribute (height)
        y_vals = [point[1] for point in cluster]  # Second attribute (weight)
        plt.scatter(x_vals, y_vals, color=colors[i % len(colors)], label=f"Cluster {i + 1}")
    
    # Plotting centroids
    centroid_x = [centroid[0] for centroid in centroids]
    centroid_y = [centroid[1] for centroid in centroids]
    plt.scatter(centroid_x, centroid_y, color='black', marker='x', s=100, label="Centroids")

    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("K-means Clustering")
    plt.legend()
    plt.show()

# Call the function to plot the clusters and centroids
plot_clusters(clusters, centroids)
