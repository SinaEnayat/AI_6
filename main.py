import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
data = iris.data

# Genetic Algorithm for Clustering
def objective_function(clustering, data):
    total_distance = 0
    for cluster_label in np.unique(clustering):
        cluster_points = data[clustering == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        distance = np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1))
        total_distance += distance
    return total_distance

def mutation(chromosome, max_clusters):
    mutation_point = np.random.randint(len(chromosome) - 1) + 1  # Avoid changing the number of clusters
    new_label = np.random.randint(1, max_clusters + 1)
    chromosome[mutation_point] = new_label
    return chromosome

def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Initialize population
def initialize_population(population_size, num_data_points, max_clusters):
    return np.random.randint(low=1, high=max_clusters + 1, size=(population_size, num_data_points))

# Main Genetic Algorithm
def genetic_algorithm(data, population_size, num_generations, max_clusters):
    num_data_points = len(data)

    # Initialize population
    population = initialize_population(population_size, num_data_points, max_clusters)

    for generation in range(num_generations):
        # Calculate fitness scores
        fitness_scores = [objective_function(chromosome, data) for chromosome in population]

        # Select parents
        parents_indices = np.argsort(fitness_scores)[:2]
        parent1, parent2 = population[parents_indices]

        # Perform Crossover and Mutation
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1, max_clusters)
        child2 = mutation(child2, max_clusters)

        # Replace low-fitness chromosomes with children
        min_fitness_index = np.argmin(fitness_scores)
        population[min_fitness_index] = child1
        population[(min_fitness_index + 1) % population_size] = child2

    # Return the best chromosome (clustering)
    best_index = np.argmin(fitness_scores)
    best_clustering = population[best_index]
    return best_clustering

# Run the Genetic Algorithm
population_size = 10
num_generations = 100
max_clusters = 3

best_clustering = genetic_algorithm(data, population_size, num_generations, max_clusters)
print("Best Clustering:", best_clustering)