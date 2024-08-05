import numpy as np
import matplotlib.pyplot as plt

# Define the objective function to be optimized
def objective_function(x):
    # Example objective function
    return x[0]**2 + x[1]**2

# Ant Colony Optimization Algorithm with convergence time optimization
def ant_colony_optimization(obj_func, num_dimensions, num_ants, num_iterations, pheromone_init, alpha, beta, rho_init, q0):
    # Initialization
    pheromone = np.full((num_dimensions, num_dimensions), pheromone_init)
    best_solution = np.zeros(num_dimensions)
    best_fitness = float('inf')  # Initialize with a large value as we are minimizing

    # Convergence-related parameters
    convergence_threshold = 1e-6
    convergence_count = 0
    max_convergence_count = 10
    rho = rho_init

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
                    probabilities[i] = (pheromone[current_node, next_node] ** alpha) * ((1 / obj_func([current_node, next_node])) ** beta)

                # Choose next node based on probability or exploit
                if np.random.random() < q0:
                    next_node = np.argmax(probabilities)
                else:
                    if np.all(probabilities == 0):
                        next_node = np.random.choice(range(len(allowed_nodes)))
                    else:
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

        # Check for convergence
        if np.all(np.abs(fitness_values - best_fitness) < convergence_threshold):
            convergence_count += 1
            if convergence_count >= max_convergence_count:
                break  # Exit loop if convergence criteria met for consecutive iterations

        # Adjust pheromone evaporation rate based on convergence
        if convergence_count >= max_convergence_count // 2:
            rho *= 0.9  # Reduce evaporation rate to enhance exploitation

        # Evaporate pheromone
        pheromone *= (1 - rho)

    return best_solution, best_fitness, iteration + 1


# Particle class
class Particle:
    def __init__(self, num_dimensions):
        self.position = np.random.rand(num_dimensions)  # Initialize random position
        self.velocity = np.random.rand(num_dimensions)  # Initialize random velocity
        self.best_position = self.position.copy()  # Initialize personal best position
        self.best_fitness = objective_function(self.position)  # Initialize personal best fitness

# Particle Swarm Optimization Algorithm with convergence time optimization
def particle_swarm_optimization(obj_func, num_dimensions, num_particles, num_iterations, convergence_threshold):
    particles = [Particle(num_dimensions) for _ in range(num_particles)]  # Initialize particles
    global_best_position = np.zeros(num_dimensions)  # Global best position
    global_best_fitness = float('inf')  # Global best fitness

    # Main loop
    for iteration in range(num_iterations):
        for particle in particles:
            # Update velocity
            inertia_weight = 0.5 + 0.5 * (num_iterations - iteration) / num_iterations  # Linearly decreasing inertia weight
            cognitive_weight = 1.5  # Cognitive weight
            social_weight = 1.5  # Social weight

            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_weight * np.random.rand(num_dimensions) * (particle.best_position - particle.position) +
                                 social_weight * np.random.rand(num_dimensions) * (global_best_position - particle.position))

            # Update position
            particle.position = np.clip(particle.position + particle.velocity, lb, ub)

            # Update personal best
            fitness = obj_func(particle.position)
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


# Individual class representing a solution (chromosome)
class Individual:
    def __init__(self, num_dimensions):
        self.chromosome = np.random.rand(num_dimensions)  # Initialize random chromosome

# Initialize the population
def initialize_population(pop_size, num_dimensions):
    return [Individual(num_dimensions) for _ in range(pop_size)]

# Perform uniform crossover between two parents
def crossover(parent1, parent2):
    child_chromosome = np.where(np.random.rand(len(parent1.chromosome)) < 0.5, parent1.chromosome, parent2.chromosome)
    child = Individual(len(parent1.chromosome))
    child.chromosome = child_chromosome
    return child

# Perform mutation on an individual
def mutate(individual, mutation_rate):
    for i in range(len(individual.chromosome)):
        if np.random.rand() < mutation_rate:
            individual.chromosome[i] = np.random.rand()
    return individual

# Select parents using tournament selection
def select_parents(population):
    parent1 = population[np.random.randint(len(population))]
    parent2 = population[np.random.randint(len(population))]
    return parent1, parent2

# Select survivors for the next generation
def select_survivors(population, offspring):
    combined_population = population + offspring
    combined_population.sort(key=lambda x: objective_function(x.chromosome))
    return combined_population[:len(population)]

# Genetic Algorithm with convergence time optimization
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


def grasshopper_optimization_algorithm(obj_func, num_dimensions, max_iterations, num_grasshoppers, lb, ub, convergence_threshold):
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
            fitness[i] = obj_func(positions[i])
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
    num_dimensions = 2
    max_iterations = 100
    num_grasshoppers = 50
    lb = [-5, 5]
    ub = [5, 5]
    convergence_threshold = 1e-6

    # Run optimization algorithms and collect convergence time data
    grasshopper_convergence_times = []
    genetic_convergence_times = []
    particle_swarm_convergence_times = []
    ant_colony_convergence_times = []

    for _ in range(10):  # Run each algorithm 10 times
        # Grasshopper Optimization Algorithm
        best_solution, best_fitness, convergence_time = grasshopper_optimization_algorithm(objective_function, num_dimensions, max_iterations, num_grasshoppers, lb, ub, convergence_threshold)
        grasshopper_convergence_times.append(convergence_time)

        # Genetic Algorithm
        pop_size = 50
        num_generations = 100
        crossover_rate = 0.8
        mutation_rate = 0.1
        best_solution, best_fitness, convergence_time = genetic_algorithm(pop_size, num_dimensions, num_generations, crossover_rate, mutation_rate, convergence_threshold)
        genetic_convergence_times.append(convergence_time)

        # Particle Swarm Optimization Algorithm
        num_particles = 50
        num_iterations = 100
        best_solution, best_fitness, convergence_time = particle_swarm_optimization(objective_function, num_dimensions, num_particles, num_iterations, convergence_threshold)
        particle_swarm_convergence_times.append(convergence_time)

        # Ant Colony Optimization Algorithm
        num_ants = 50
        num_iterations = 100
        pheromone_init = 1.0
        alpha = 1.0
        beta = 1.0
        rho_init = 0.1
        q0 = 0.9
        best_solution, best_fitness, convergence_time = ant_colony_optimization(objective_function, num_dimensions, num_ants, num_iterations, pheromone_init, alpha, beta, rho_init, q0)
        ant_colony_convergence_times.append(convergence_time)

    # Visualize convergence time data
    algorithms = ['GOA', 'GA', 'PSO', 'ACO']
    grasshopper_convergence_times = [10, 15, 20, 25]  # Example data, replace with actual convergence times
    genetic_convergence_times = [12, 18, 22, 27]  # Example data, replace with actual convergence times
    particle_swarm_convergence_times = [14, 16, 25, 20]  # Example data, replace with actual convergence times
    ant_colony_convergence_times = [13, 17, 19, 23]  # Example data, replace with actual convergence times

    # Combine convergence times
    convergence_times = [grasshopper_convergence_times, genetic_convergence_times, particle_swarm_convergence_times, ant_colony_convergence_times]

    # Plotting
    plt.figure(figsize=(5, 6))
    plt.bar(np.arange(len(algorithms)), [np.mean(times) for times in convergence_times], yerr=[np.std(times) for times in convergence_times], capsize=5,color=['blue', 'green', 'orange', 'red'])
    plt.xticks(np.arange(len(algorithms)), algorithms)
    plt.ylabel('Convergence Time (second)')
    plt.title('Convergence Time comparison of Optimization Algorithms')
    plt.grid(True)
    plt.show()


