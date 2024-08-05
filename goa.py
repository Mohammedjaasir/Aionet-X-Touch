import time
import numpy as np

class Grasshopper:
    def __init__(self, num_dimensions, num_agents, max_iter):
        self.num_dimensions = num_dimensions
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.agents_position = np.random.uniform(-10, 10, (num_agents, num_dimensions))
        self.agents_velocity = np.zeros((num_agents, num_dimensions))  # Initialize velocities
        self.best_position = None
        self.best_fitness = float('inf')
        self.iteration = 0  # Initialize iteration count

    def objective_function(self, x):
        # Placeholder for the objective function
        power_consumption = 4.5
        packet_delivery_rate = 93.2
        convergence_time = 280.75
        
        # Define weights for each objective
        weight_power_consumption = 0.3  # Adjust according to importance
        weight_packet_delivery_rate = 0.4  # Adjust according to importance
        weight_convergence_time = 0.3  # Adjust according to importance
        
        # Combine objectives using weighted sum
        combined_fitness = (weight_power_consumption * power_consumption +
                            weight_packet_delivery_rate * (1 - packet_delivery_rate) +
                            weight_convergence_time * convergence_time)
        return combined_fitness

    def update_best_position(self):
        for i in range(self.num_agents):
            fitness = self.objective_function(self.agents_position[i])
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = self.agents_position[i]

    def update_agents_position(self):
        w = 0.9 - self.iteration * (0.9 / self.max_iter)  # Inertia weight decreases linearly
        c1 = 2  # Cognitive parameter
        c2 = 2  # Social parameter

        for i in range(self.num_agents):
            r1 = np.random.rand(self.num_dimensions)
            r2 = np.random.rand(self.num_dimensions)

            # Update velocity
            self.agents_velocity[i] = (w * self.agents_velocity[i] +
                                       c1 * r1 * (self.best_position - self.agents_position[i]) +
                                       c2 * r2 * (self.best_position - self.agents_position[i]))

            # Update position
            self.agents_position[i] = self.agents_position[i] + self.agents_velocity[i]

    def optimize(self):
        start_time = time.time()  # Start timer
        while self.iteration < self.max_iter:
            self.update_best_position()
            self.update_agents_position()
            self.iteration += 1
        end_time = time.time()  # Stop timer
        convergence_time = end_time - start_time
        #print(f"Convergence time: {convergence_time} seconds")
        print(f"Best fitness value: {self.best_fitness}")

# Example usage:
num_dimensions = 2
num_agents = 20
max_iter = 100
goa = Grasshopper(num_dimensions, num_agents, max_iter)
goa.optimize()
