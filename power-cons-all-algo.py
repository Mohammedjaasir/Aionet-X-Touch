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
    combined_population.sort(key=lambda x: objective_function(x.chromosome))
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

    best_solution = min(population, key=lambda x: objective_function(x.chromosome))
    best_fitness = objective_function(best_solution.chromosome)
    return best_solution.chromosome, best_fitness


# Grasshopper Optimization Algorithm with power consumption
def grasshopper_optimization_algorithm(obj_func, num_dimensions, max_iterations, num_grasshoppers, lb, ub):
    # Initialization
    positions = np.random.uniform(lb, ub, (num_grasshoppers, num_dimensions))
    fitness = np.zeros(num_grasshoppers)
    best_solution = np.zeros(num_dimensions)
    best_fitness = float('inf') # Initialize with a large value as we are minimizing power consumption

    convergence_time = 0  # Initialize convergence time

    # Main loop
    for iteration in range(max_iterations):
        # Calculate fitness for each grasshopper
        for i in range(num_grasshoppers):
            fitness[i] = obj_func(positions[i])

            # Update the global best solution
            if fitness[i] < best_fitness:
                best_fitness = fitness[i]
                best_solution = positions[i].copy()
                convergence_time = iteration + 1  # Update convergence time

        # Update positions of grasshoppers
        for i in range(num_grasshoppers):
            for j in range(num_dimensions):
                r1 = np.random.random()
                r2 = np.random.random()

                # Update position using GOA formula
                positions[i][j] = positions[i][j] + r1 * (best_solution[j] - positions[i][j]) + r2 * np.mean(positions[:, j]) - positions[i][j]

                # Boundary control
                positions[i][j] = np.clip(positions[i][j], lb[j], ub[j])

        # Check for convergence
        if np.all(np.abs(fitness - best_fitness) < 1e-6):
            break

    return best_solution, best_fitness, convergence_time


# Particle class
class Particle:
    def __init__(self, num_dimensions):
        self.position = np.random.rand(num_dimensions)  # Initialize random position
        self.velocity = np.random.rand(num_dimensions)  # Initialize random velocity
        self.best_position = self.position.copy()  # Initialize personal best position
        self.best_fitness = objective_function(self.position)  # Initialize personal best fitness

# Particle Swarm Optimization Algorithm
def particle_swarm_optimization(obj_func, num_dimensions, num_particles, num_iterations):
    particles = [Particle(num_dimensions) for _ in range(num_particles)]  # Initialize particles
    global_best_position = np.zeros(num_dimensions)  # Global best position
    global_best_fitness = float('inf')  # Global best fitness

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
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

    return global_best_position, global_best_fitness

if __name__ == "__main__":
  num_dimensions = 3
  max_iterations_goa=100
  max_iterations_pso=100
  num_iterations_aco = 100
  num_grasshoppers = 20
  num_ants = 50
  num_particles = 20
  pop_size_ga = 50
  num_generations_ga = 50
  lb = [0, 0, 0]
  ub = [10, 100, 1]
  inertia_weight = 0.5
  cognitive_weight = 1.5
  social_weight = 1.5
  crossover_rate_ga = 0.8
  mutation_rate_ga = 0.1
  pheromone_init = 1.0
  alpha = 1.0
  beta = 1.0
  rho = 0.1
  q0 = 0.9

  # Run Ant Colony Optimization algorithm
  best_solution_aco, best_fitness_aco = ant_colony_optimization(objective_function, num_dimensions, num_ants, num_iterations_aco, pheromone_init, alpha, beta, rho, q0)

  # Run Genetic Algorithm
  best_solution_ga, best_fitness_ga = genetic_algorithm(pop_size_ga, num_dimensions, num_generations_ga, crossover_rate_ga, mutation_rate_ga)

  # Run Grasshopper Optimization Algorithm
  best_solution_goa, best_fitness_goa, _ = grasshopper_optimization_algorithm(objective_function, num_dimensions, max_iterations_goa, num_grasshoppers, lb, ub)

  # Run Particle Swarm Optimization algorithm
  best_solution_pso, best_fitness_pso = particle_swarm_optimization(objective_function, num_dimensions, num_particles, max_iterations_pso)

  algorithms = ['ACO', 'GA', 'GOA', 'PSO']
  power_consumptions = [best_fitness_aco, best_fitness_ga, best_fitness_goa, best_fitness_pso]
  import matplotlib.pyplot as plt

  plt.figure(figsize=(5, 6))
  plt.bar(algorithms, power_consumptions, color=['blue', 'green', 'orange', 'red'])
  plt.xlabel('Optimization Algorithm')
  plt.ylabel('Power Consumption(seconds)')
  plt.title('Comparison of Power Consumption Across Optimization Algorithms')
  plt.show()





