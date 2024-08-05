import numpy as np

# Define the objective function to be optimized (Packet Delivery Rate)
def objective_function(x):
  x=[1,2,3,4]
  if len(x) == 3:
        raise ValueError("Input array 'x' must have at least 3 elements.")

    # Example objective function for packet delivery rate optimization
    # Here, network_config represents the parameters of the network configuration
    # You need to replace this with your actual objective function
    # This function should take network configuration parameters as input and return the packet delivery rate
    # For demonstration purposes, let's assume a simple linear relationship between parameters and packet delivery rate
  return 0.5 * x[0] + 0.3 * x[1] + 0.2 * x[2]

# Ant Colony Optimization Algorithm for packet delivery rate optimization
def ant_colony_optimization_packet_delivery_rate(num_ants, num_iterations, num_dimensions, alpha, beta, rho, q0, lb, ub):
    num_nodes = num_dimensions
    pheromone = np.full((num_nodes, num_nodes), 1e-6)  # Initialize pheromone levels
    best_solution = None
    best_fitness = float('-inf')

    for _ in range(num_iterations):
        solutions = np.zeros((num_ants, num_dimensions), dtype=int)
        fitness_values = np.zeros(num_ants)

        for ant in range(num_ants):
            current_node = np.random.randint(num_nodes)
            visited_nodes = [current_node]

            for _ in range(num_dimensions - 1):
                allowed_nodes = np.setdiff1d(np.arange(num_nodes), visited_nodes)
                probabilities = np.zeros(len(allowed_nodes))

                for i, next_node in enumerate(allowed_nodes):
                    pheromone_factor = pheromone[current_node, next_node] ** alpha
                    heuristic_factor = (1 / objective_function(allowed_nodes)) ** beta
                    probabilities[i] = pheromone_factor * heuristic_factor

                if np.random.rand() < q0:
                    next_node = np.argmax(probabilities)
                else:
                    probabilities /= np.sum(probabilities)
                    next_node = np.random.choice(range(len(allowed_nodes)), p=probabilities)

                solutions[ant, _] = allowed_nodes[next_node]
                visited_nodes.append(allowed_nodes[next_node])
                current_node = allowed_nodes[next_node]

            if len(visited_nodes) < num_nodes:
                # There are unvisited nodes
                solutions[ant, -1] = np.setdiff1d(np.arange(num_nodes), visited_nodes)[0]
            else:
                # All nodes have been visited
                # For example, you can set the last node to be the same as the first node to complete the tour
                solutions[ant, -1] = solutions[ant, 0]
            fitness_values[ant] = objective_function(solutions[ant])

        if np.max(fitness_values) > best_fitness:
            best_solution = solutions[np.argmax(fitness_values)]
            best_fitness = np.max(fitness_values)

        delta_pheromone = np.zeros((num_nodes, num_nodes))

        for ant in range(num_ants):
            for i in range(num_dimensions - 1):
                delta_pheromone[solutions[ant, i], solutions[ant, i+1]] += 1 / fitness_values[ant]

        pheromone = (1 - rho) * pheromone + delta_pheromone

    return best_solution, best_fitness

# Define the Individual class for Genetic Algorithm
class Individual:
    def __init__(self, num_dimensions):
        self.chromosome = np.random.rand(num_dimensions)  # Initialize random chromosome

# Initialize the population for Genetic Algorithm
def initialize_population(pop_size, num_dimensions):
    return [Individual(num_dimensions) for _ in range(pop_size)]

# Perform uniform crossover between two parents for Genetic Algorithm
def crossover(parent1, parent2):
    child_chromosome = np.where(np.random.rand(len(parent1.chromosome)) < 0.5, parent1.chromosome, parent2.chromosome)
    child = Individual(len(parent1.chromosome))
    child.chromosome = child_chromosome
    return child

# Perform mutation on an individual for Genetic Algorithm
def mutate(individual, mutation_rate):
    for i in range(len(individual.chromosome)):
        if np.random.rand() < mutation_rate:
            individual.chromosome[i] = np.random.rand()
    return individual

# Select parents using tournament selection for Genetic Algorithm
def select_parents(population):
    parent1 = population[np.random.randint(len(population))]
    parent2 = population[np.random.randint(len(population))]
    return parent1, parent2

# Select survivors for the next generation for Genetic Algorithm
def select_survivors(population, offspring):
    combined_population = population + offspring
    combined_population.sort(key=lambda x: objective_function(x.chromosome))
    return combined_population[:len(population)]

# Genetic Algorithm for packet delivery rate optimization
def genetic_algorithm(pop_size, num_dimensions, num_generations, crossover_rate, mutation_rate, convergence_threshold):
    population = initialize_population(pop_size, num_dimensions)
    convergence_time = 0

    for generation in range(num_generations):
        offspring = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population)
            if np.random.rand() < crossover_rate:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
            else:
                child1 = parent1
                child2 = parent2
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            offspring.append(child1)
            offspring.append(child2)
        population = select_survivors(population, offspring)

        # Check convergence
        if objective_function(population[0].chromosome) < convergence_threshold:
            convergence_time = generation + 1
            break

    best_solution = min(population, key=lambda x: objective_function(x.chromosome))
    best_fitness = objective_function(best_solution.chromosome)
    return best_solution.chromosome, best_fitness, convergence_time

# Define the Particle class for Particle Swarm Optimization
class Particle:
    def __init__(self, num_dimensions):
        self.position = np.random.rand(num_dimensions)  # Initialize random position
        self.velocity = np.random.rand(num_dimensions)  # Initialize random velocity
        self.best_position = self.position.copy()  # Initialize personal best position
        self.best_fitness = objective_function(self.position)  # Initialize personal best fitness

# Particle Swarm Optimization Algorithm for packet delivery rate optimization
def particle_swarm_optimization(num_dimensions, num_particles, num_iterations, convergence_threshold):
    particles = [Particle(num_dimensions) for _ in range(num_particles)]  # Initialize particles
    global_best_position = np.zeros(num_dimensions)  # Global best position
    global_best_fitness = float('inf')  # Global best fitness

    # Main loop
    for iteration in range(num_iterations):
        for particle in particles:
            # Update velocity
            cognitive_weight = 1.5  # Cognitive weight
            social_weight = 1.5  # Social weight

            particle.velocity = (particle.velocity +
                                 cognitive_weight * np.random.rand(num_dimensions) * (particle.best_position - particle.position) +
                                 social_weight * np.random.rand(num_dimensions) * (global_best_position - particle.position))

            # Update position
            particle.position = np.clip(particle.position + particle.velocity, 0, 1)

            # Update personal best
            fitness = objective_function(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        # Check convergence
        if global_best_fitness < convergence_threshold:
            break

    return global_best_position, global_best_fitness, iteration + 1

# Grasshopper Optimization Algorithm for packet delivery rate optimization
def grasshopper_optimization_algorithm(num_dimensions, max_iterations, num_grasshoppers, lb, ub, convergence_threshold):
    # Initialization
    positions = np.random.uniform(lb, ub, (num_grasshoppers, num_dimensions))
    fitness = np.zeros(num_grasshoppers)
    best_solution = np.zeros(num_dimensions)
    best_fitness = float('inf')
    convergence_time = 0

    # Main loop
    for iteration in range(max_iterations):
        # Calculate fitness for each grasshopper
        for i in range(num_grasshoppers):
            fitness[i] = objective_function(positions[i])
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_solution = positions[i].copy()
                convergence_time = iteration + 1

        # Update positions of grasshoppers
        for i in range(num_grasshoppers):
            for j in range(num_dimensions):
                r1 = np.random.random()
                r2 = np.random.random()
                positions[i, j] = positions[i, j] + r1 * (best_solution[j] - positions[i, j]) + r2 * np.mean(positions[:, j]) - positions[i, j]
                positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])

        # Check convergence
        if np.all(np.abs(fitness - best_fitness) < convergence_threshold):
            break

    return best_solution, best_fitness, convergence_time


if __name__ == "__main__":
    # Define optimization parameters
    num_dimensions = 3  # Number of network configuration parameters
    num_ants = 60
    num_iterations = 100
    pheromone_init = 1.0
    alpha = 1.0
    beta = 1.0
    rho_init = 0.1
    q0 = 0.9
    lb = [0, 0, 0]  # Lower bounds for network configuration parameters
    ub = [10, 10, 10]  # Upper bounds for network configuration parameters

    pop_size = 70
    num_generations = 100
    crossover_rate = 1.0
    mutation_rate = 0.1
    convergence_threshold = 1e-7

    num_particles = 50

    max_iterations = 100
    num_grasshoppers = 80
    convergence_threshold = 1e-6

    # Run optimization algorithms
    best_solution_aco, best_fitness_aco = ant_colony_optimization_packet_delivery_rate(num_ants, num_iterations, num_dimensions, alpha, beta, rho_init, q0, lb, ub)
    best_solution_ga, best_fitness_ga, convergence_time_ga = genetic_algorithm(pop_size, num_dimensions, num_generations, crossover_rate, mutation_rate, convergence_threshold)
    best_solution_pso, best_fitness_pso, convergence_time_pso = particle_swarm_optimization(num_dimensions, num_particles, num_iterations, convergence_threshold)
    best_solution_go, best_fitness_go, convergence_time_go = grasshopper_optimization_algorithm(num_dimensions, max_iterations, num_grasshoppers, lb, ub, convergence_threshold)

    # Print results
    print("Ant Colony Optimization - Packet Delivery Rate:", best_fitness_aco/num_ants)
    print("Genetic Algorithm - Packet Delivery Rate:", best_fitness_ga/pop_size)
    print("Particle Swarm Optimization - Packet Delivery Rate:", best_fitness_pso/num_particles)
    print("Grasshopper Optimization - Packet Delivery Rate:", best_fitness_go/num_grasshoppers)



    import matplotlib.pyplot as plt

    # Results from optimization algorithms
    optimization_algorithms = ["ACO", "GA", "PSO", "GAO"]
    packet_delivery_rates = [best_fitness_aco/num_ants, best_fitness_ga/pop_size, best_fitness_pso/num_particles, best_fitness_go/num_grasshoppers]

    # Plotting
    plt.figure(figsize=(5, 6))
    plt.bar(optimization_algorithms, packet_delivery_rates, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Optimization Algorithm')
    plt.ylabel('Packet Delivery Rate(seconds)')
    plt.title('Comparison of Optimization Algorithms for Packet Delivery Rate')
    plt.ylim(0, max(packet_delivery_rates) * 1.1)  # Adjust ylim for better visualization
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


