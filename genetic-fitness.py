import numpy as np

class Individual:
    def __init__(self, num_dimensions):
        self.chromosome = np.random.rand(num_dimensions)  # Initialize random chromosome

def objective_function(x):
    # Assuming x is a vector containing values for convergence time, power consumption, and packet delivery rate
    # You may need to normalize these values before using them in the objective function
    # Here, we simply sum the inverse of convergence time, power consumption, and packet delivery rate
    # The goal is to maximize convergence time and packet delivery rate while minimizing power consumption
    return 1 / x[0] + 1 / x[1] + x[2]

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

# Example usage
if __name__ == "__main__":
    pop_size = 50
    num_dimensions = 3  # Convergence time, power consumption, packet delivery rate
    num_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.1

    best_solution, best_fitness = genetic_algorithm(pop_size, num_dimensions, num_generations, crossover_rate, mutation_rate)

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)

