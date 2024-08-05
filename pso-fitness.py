import numpy as np

class Particle:
    def __init__(self, num_dimensions):
        self.position = np.random.rand(num_dimensions)  # Initialize random position
        self.velocity = np.random.rand(num_dimensions)  # Initialize random velocity
        self.best_position = self.position.copy()  # Personal best position
        self.best_fitness = float('-inf')  # Personal best fitness

def objective_function(x):
    # Assuming x is a vector containing values for convergence time, power consumption, and packet delivery rate
    # You may need to normalize these values before using them in the objective function
    # Here, we simply sum the inverse of convergence time, power consumption, and packet delivery rate
    # The goal is to maximize convergence time and packet delivery rate while minimizing power consumption
    return 1 / x[0] + 1 / x[1] + x[2]

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

# Example usage
if __name__ == "__main__":
    num_dimensions = 3  # Convergence time, power consumption, packet delivery rate
    num_particles = 20
    num_iterations = 100
    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5
    lb = np.array([0, 0, 0])  # Lower bounds for convergence time, power consumption, and packet delivery rate
    ub = np.array([10, 100, 1])  # Upper bounds for convergence time, power consumption, and packet delivery rate

    best_solution, best_fitness = particle_swarm_optimization(objective_function, num_dimensions, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight, lb, ub)

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)

