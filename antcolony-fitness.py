import numpy as np

# Define the objective function to be optimized (representing power consumption)
def objective_function(x):
    # Example: Power consumption = f(x) = 1 / (x[0] + x[1] + x[2])
    if x[0] == 0 or x[1] == 0:
        # Return a large value to discourage selection of this solution
        return float('inf')
    return 1 / (x[0] + x[1] + x[2])

# Ant Colony Optimization Algorithm
def ant_colony_optimization(obj_func, num_dimensions, num_ants, num_iterations, pheromone_init, alpha, beta, rho, q0):
    # Initialization
    pheromone = np.full((num_dimensions, num_dimensions), pheromone_init)
    best_solution = np.zeros(num_dimensions)
    best_fitness = float('inf')  # Initialize with a large value as we are minimizing power consumption

    # Main loop
    for iteration in range(num_iterations):
        solutions = np.zeros((num_ants, num_dimensions))
        fitness_values = np.zeros(num_ants)

        # Construct solutions for each ant
        for ant in range(num_ants):
            current_node = np.random.randint(num_dimensions)  # Start from a random node

            # Construct solution path for the current ant
            for step in range(num_dimensions - 1):
                allowed_nodes = np.delete(np.arange(num_dimensions), current_node)
                probabilities = np.zeros(len(allowed_nodes))

                # Calculate probabilities for selecting next node based on pheromone levels and heuristic information
                for i, next_node in enumerate(allowed_nodes):
                    probabilities[i] = (pheromone[current_node, next_node] ** alpha) * ((1 / obj_func([current_node, next_node, 0])) ** beta)

                # Check for NaN values in probabilities
                if np.any(np.isnan(probabilities)):
                    # If probabilities contain NaN, set them to zero
                    probabilities[np.isnan(probabilities)] = 0

                # Choose next node based on probability or exploit
                if np.random.random() < q0:
                    next_node = np.argmax(probabilities)
                else:
                    if np.all(probabilities == 0):
                        # If all probabilities are zero, choose randomly
                        next_node = np.random.choice(range(len(allowed_nodes)))
                    else:
                        # Normalize probabilities to avoid division by zero
                        probabilities /= np.sum(probabilities)
                        next_node = np.random.choice(range(len(allowed_nodes)), p=probabilities)

                solutions[ant, step] = current_node
                current_node = allowed_nodes[next_node]

            # Add the last node to complete the solution
            solutions[ant, -1] = current_node

            # Calculate fitness for the solution
            fitness_values[ant] = obj_func(solutions[ant])

        # Update pheromone levels
        for i in range(num_ants):
            for j in range(num_dimensions - 1):
                pheromone[int(solutions[i, j]), int(solutions[i, j + 1])] += 1 / fitness_values[i]

        # Update global best solution
        current_best_index = np.argmin(fitness_values)
        if fitness_values[current_best_index] < best_fitness:
            best_fitness = fitness_values[current_best_index]
            best_solution = solutions[current_best_index]

        # Evaporate pheromone
        pheromone *= (1 - rho)

    return best_solution, best_fitness

# Example usage
if __name__ == "__main__":
    num_dimensions = 3
    num_ants = 50
    num_iterations = 100
    pheromone_init = 1.0
    alpha = 1.0
    beta = 1.0
    rho = 0.1
    q0 = 0.9

    # Run ACO algorithm
    best_solution, best_fitness = ant_colony_optimization(objective_function, num_dimensions, num_ants, num_iterations, pheromone_init, alpha, beta, rho, q0)

    #print("Best Solution:", best_solution)
    aco_power=best_fitness

