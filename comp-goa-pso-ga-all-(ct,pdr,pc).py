import numpy as np
import matplotlib.pyplot as plt

# Define the objective function to be optimized
def objective_function(x):
  # Check if any of the denominators are zero
    if x[0] == 0 or x[1] == 0:
        # Return a large negative value to discourage selection of this solution
        return float('-inf')
    return 1 / x[0] + 1 / x[1] + x[2]
# Grasshopper Optimization Algorithm
def grasshopper_optimization_algorithm(obj_func, num_dimensions, num_iterations, num_grasshoppers, lb, ub):
    # Initialization
    positions = np.random.uniform(lb, ub, (num_grasshoppers, num_dimensions))
    fitness = np.zeros(num_grasshoppers)
    best_solution = np.zeros(num_dimensions)
    best_fitness = float('-inf') # Negative infinity as we are maximizing the objective function

    # Main loop
    for iteration in range(num_iterations):
        # Calculate fitness for each grasshopper
        for i in range(num_grasshoppers):
            fitness[i] = obj_func(positions[i])

            # Update the global best solution
            if fitness[i] > best_fitness:
                best_fitness = fitness[i]
                best_solution = positions[i].copy()

        # Update positions of grasshoppers
        for i in range(num_grasshoppers):
            for j in range(num_dimensions):
                r1 = np.random.random()
                r2 = np.random.random()

                # Update position using GOA formula
                positions[i][j] = positions[i][j] + r1 * (best_solution[j] - positions[i][j]) + r2 * np.mean(positions[:, j]) - positions[i][j]

                # Boundary control
                positions[i][j] = np.clip(positions[i][j], lb[j], ub[j])

    return best_solution, best_fitness

# Ant Colony Optimization Algorithm
def ant_colony_optimization(obj_func, num_dimensions, num_ants, num_iterations, pheromone_init, alpha, beta, rho, q0):
    # Initialization
    pheromone = np.full((num_dimensions, num_dimensions), pheromone_init)
    best_solution = np.zeros(num_dimensions)
    best_fitness = float('-inf') # Negative infinity as we are maximizing the objective function

    # Main loop
    for iteration in range(num_iterations):
        solutions = np.zeros((num_ants, num_dimensions))
        fitness_values = np.zeros(num_ants)

        # Construct solutions for each ant
        for ant in range(num_ants):
            current_node = np.random.randint(num_dimensions) # Start from a random node

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
        current_best_index = np.argmax(fitness_values)
        if fitness_values[current_best_index] > best_fitness:
            best_fitness = fitness_values[current_best_index]
            best_solution = solutions[current_best_index]

        # Evaporate pheromone
        pheromone *= (1 - rho)

    return best_solution, best_fitness

# Particle Swarm Optimization Algorithm
def particle_swarm_optimization(obj_func, num_dimensions, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight, lb, ub):
    particles = [Particle(num_dimensions) for _ in range(num_particles)]  # Initialize particles
    global_best_position = np.zeros(num_dimensions)  # Global best position
    global_best_fitness = float('-inf')  # Global best fitness

    for iteration in range(num_iterations):
        for particle in particles:
            # Update velocity
            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_weight * np.random.rand(num_dimensions) * (particle.best_position - particle.position) +
                                 social_weight * np.random.rand(num_dimensions) * (global_best_position - particle.position))

            # Update position
            particle.position = np.clip(particle.position + particle.velocity, lb, ub)

            # Update personal best
            fitness = obj_func(particle.position)
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

    return global_best_position, global_best_fitness

# Genetic Algorithm
class Individual:
    def __init__(self, num_dimensions):
        self.chromosome = np.random.rand(num_dimensions)  # Initialize random chromosome

def initialize_population(pop_size, num_dimensions):
    return [Individual(num_dimensions) for _ in range(pop_size)]

def crossover(parent1, parent2):
    # Uniform crossover
    child_chromosome = np.where(np.random.rand(len(parent1.chromosome)) < 0.5, parent1.chromosome, parent2.chromosome)
    child = Individual(len(parent1.chromosome))
    child.chromosome = child_chromosome
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual.chromosome)):
        if np.random.rand() < mutation_rate:
            individual.chromosome[i] = np.random.rand()
    return individual

def select_parents(population):
    # Tournament selection
    parent1 = population[np.random.randint(len(population))]
    parent2 = population[np.random.randint(len(population))]
    return parent1, parent2

def select_survivors(population, offspring):
    # Elitism: Keep the best individuals from the current population and offspring
    combined_population = population + offspring
    combined_population.sort(key=lambda x: objective_function(x.chromosome), reverse=True)
    return combined_population[:len(population)]

def genetic_algorithm(pop_size, num_dimensions, num_generations, crossover_rate, mutation_rate):
    population = initialize_population(pop_size, num_dimensions)

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

    best_solution = max(population, key=lambda x: objective_function(x.chromosome))
    best_fitness = objective_function(best_solution.chromosome)
    return best_solution.chromosome, best_fitness


# Function to run an optimization algorithm multiple times and collect the best fitness values
def run_algorithm(algorithm, *args, num_runs=50):
    best_fitness_values = []
    for _ in range(num_runs):
        _, best_fitness = algorithm(*args)
        best_fitness_values.append(best_fitness)
    return best_fitness_values

# Example usage
if __name__ == "__main__":
    num_dimensions = 3
    num_iterations = 100
    num_grasshoppers = 20
    num_ants = 50
    num_particles = 20
    pop_size = 50
    num_generations = 50
    lb = [0, 0, 0]
    ub = [10, 100, 1]
    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5
    crossover_rate = 0.8
    mutation_rate = 0.1
    pheromone_init = 1.0
    alpha = 1.0
    beta = 1.0
    rho = 0.1
    q0 = 0.9

    # Run each algorithm multiple times and collect the best fitness values
    goa_fitness = run_algorithm(grasshopper_optimization_algorithm, objective_function, num_dimensions, num_iterations, num_grasshoppers, lb, ub)
    aco_fitness = run_algorithm(ant_colony_optimization, objective_function, num_dimensions, num_ants, num_iterations, 1.0, 1.0, 1.0, 0.1, 0.9)
    pso_fitness = run_algorithm(particle_swarm_optimization, objective_function, num_dimensions, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight, lb, ub)
    ga_fitness = run_algorithm(genetic_algorithm, pop_size, num_dimensions, num_generations, crossover_rate, mutation_rate)

    # Plot histograms
    #plt.hist(goa_fitness, bins=20, alpha=0.5, label='GOA')
    #plt.hist(aco_fitness, bins=20, alpha=0.5, label='ACO')
    #plt.hist(pso_fitness, bins=20, alpha=0.5, label='PSO')
    #plt.hist(ga_fitness, bins=20, alpha=0.5, label='GA')
    #plt.xlabel('Fitness')
    #plt.ylabel('Frequency')
    #plt.title('Comparison of Optimization Algorithms')
    #plt.legend()
    #plt.show()
    #algorithms = ['GOA', 'ACO', 'PSO', 'GA']
    #mean_fitness = [np.mean(goa_fitness), np.mean(aco_fitness), np.mean(pso_fitness), np.mean(ga_fitness)]
    #std_dev = [np.std(goa_fitness), np.std(aco_fitness), np.std(pso_fitness), np.std(ga_fitness)]
    """plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(goa_fitness, bins=20, alpha=0.5, color='blue', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('GOA')


    plt.subplot(2, 2, 2)
    plt.hist(aco_fitness, bins=20, alpha=0.5, color='green', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('ACO')


    plt.subplot(2, 2, 3)
    plt.hist(pso_fitness, bins=20, alpha=0.5, color='orange', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('PSO')


    plt.subplot(2, 2, 4)
    plt.hist(ga_fitness, bins=20, alpha=0.5, color='red', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('GA')

    plt.tight_layout()
    plt.show()



    # Plot bar graphs
    #plt.bar(algorithms, mean_fitness, yerr=std_dev, capsize=5)
    #plt.xlabel('Algorithm')
    #plt.ylabel('Mean Fitness')
    #plt.title('Comparison of Optimization Algorithms')
    #plt.show()

    plt.figure(figsize=(10, 6))
    algorithms = ['GOA', 'ACO', 'PSO', 'GA']
    all_fitness = [goa_fitness, aco_fitness, pso_fitness, ga_fitness]
    colors = ['blue', 'green', 'orange', 'red']

    for i, (algo, fitness) in enumerate(zip(algorithms, all_fitness)):
        plt.hist(fitness, bins=20, alpha=0.5, color=colors[i], label=algo)

    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('OverallComparison of Optimization Algorithms')
    plt.legend()
    plt.show()"""

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    algorithms = ['GOA', 'ACO', 'PSO', 'GA']
    all_fitness = [goa_fitness, aco_fitness, pso_fitness, ga_fitness]
    dimensions = ['Convergence Time', 'Power Consumption', 'Packet Delivery Rate']
    colors = ['blue', 'green', 'orange', 'red']

    for dim, ax in zip(dimensions, axs):
      for algo, fitness, color in zip(algorithms, all_fitness, colors):
        ax.hist(fitness, bins=20, alpha=0.5, color=color, label=algo)
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Comparison of Optimization Algorithms - {dim}')
        ax.legend()

    plt.tight_layout()
    plt.show()


