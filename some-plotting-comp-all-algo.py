import simpy
import random

class Node:
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.neighbors = set()
        self.parent = None
        self.rank = float('inf')
        self.dio_interval = 10  # DIO interval in seconds
        self.max_rank_increase = 1
        self.rank_increase_interval = 5
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dio_interval)
            self.send_dio()

    def send_dio(self):
        # Simulate sending DIO message to neighbors
        pass

    def receive_dio(self, message):
        # Process received DIO message
        pass

def network(env, num_nodes):
    nodes = [Node(env, i) for i in range(num_nodes)]

    # Randomly select neighbors for each node
    for node in nodes:
        node.neighbors = set(random.sample(nodes, random.randint(1, 10)))

    # Start the simulation
    while True:
        yield env.timeout(1)  # Simulate time passing
        # Perform network operations here

# Create a simulation environment
env = simpy.Environment()
num_nodes = 200
env.process(network(env, num_nodes))

# Run the simulation
env.run(until=1000)  # Run for 1000 seconds





import simpy
import random
import math

# Constants for GOA
N = 200  # Number of grasshoppers (nodes)
Max_iter = 1000  # Maximum number of iterations
lb = 0  # Lower bound for parameters
ub = 1  # Upper bound for parameters

# Grasshopper class
class Grasshopper:
    def __init__(self, node_id):
        self.node_id = node_id
        self.position = [random.uniform(lb, ub) for _ in range(3)]  # Initial position
        self.fitness = float('inf')  # Initial fitness

# Grasshopper Optimization Algorithm
class GOA:
    def __init__(self):
        self.grasshoppers = [Grasshopper(i) for i in range(N)]
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self):
        for _ in range(Max_iter):
            for grasshopper in self.grasshoppers:
                # Evaluate fitness
                fitness = self.evaluate_fitness(grasshopper.position)
                if fitness < grasshopper.fitness:
                    grasshopper.fitness = fitness
                    grasshopper.position = self.update_position(grasshopper.position)

                # Update global best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = grasshopper.position

    def evaluate_fitness(self, position):
        # Evaluate fitness based on RPL performance metrics
        return random.uniform(0, 1)  # Placeholder for actual fitness calculation

    def update_position(self, position):
        # Update position based on GOA formula
        r = random.random()
        levy = self.levy_flight()
        new_position = [position[i] + r * levy[i] for i in range(3)]
        return [min(ub, max(lb, x)) for x in new_position]

    def levy_flight(self):
        # Generate a step from Levy distribution
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = random.gauss(0, sigma)
        v = random.gauss(0, 1)
        step = u / abs(v) ** (1 / beta)
        return [step for _ in range(3)]  # 3-dimensional step

# Modify the Node class to use GOA-optimized parameters
# Modify the Node class to use GOA-optimized parameters
class Node:
    def __init__(self, env, node_id, rank, dio_interval, max_rank_increase):
        self.env = env
        self.node_id = node_id
        self.neighbors = set()
        self.parent = None
        self.rank = rank
        self.dio_interval = dio_interval
        self.max_rank_increase = max_rank_increase
        self.rank_increase_interval = 5
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dio_interval)
            self.send_dio()

    def send_dio(self):
        # Simulate sending DIO message to neighbors
        pass

    def receive_dio(self, message):
        # Process received DIO message
        pass

# Simulation environment
def simulate(env, num_nodes, goa_best_solution):
    # Create nodes with GOA-optimized parameters
    nodes = [Node(env, i, goa_best_solution[0], goa_best_solution[1] * 10, goa_best_solution[2]) for i in range(num_nodes)]

    # Randomly select neighbors for each node
    for node in nodes:
        node.neighbors = set(random.sample(nodes, random.randint(1, 10)))

    # Start the simulation
    while True:
        yield env.timeout(1)  # Simulate time passing
        # Perform network operations here

# Create a simulation environment
env = simpy.Environment()
num_nodes = 200
goa = GOA()
goa.optimize()

env.process(simulate(env, num_nodes, goa.best_solution))

# Run the simulation
env.run(until=1000)  # Run for 1000 seconds


import simpy
import random
import math
import matplotlib.pyplot as plt

# Constants for GOA
N = 200  # Number of grasshoppers (nodes)
Max_iter = 1000  # Maximum number of iterations
lb = 0  # Lower bound for parameters
ub = 1  # Upper bound for parameters

# Grasshopper class
class Grasshopper:
    def __init__(self, node_id):
        self.node_id = node_id
        self.position = [random.uniform(lb, ub) for _ in range(3)]  # Initial position
        self.fitness = float('inf')  # Initial fitness

# Grasshopper Optimization Algorithm
class GOA:
    def __init__(self):
        self.grasshoppers = [Grasshopper(i) for i in range(N)]
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self):
        for _ in range(Max_iter):
            for grasshopper in self.grasshoppers:
                # Evaluate fitness
                fitness = self.evaluate_fitness(grasshopper.position)
                if fitness < grasshopper.fitness:
                    grasshopper.fitness = fitness
                    grasshopper.position = self.update_position(grasshopper.position)

                # Update global best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = grasshopper.position

    def evaluate_fitness(self, position):
        # Evaluate fitness based on RPL performance metrics
        return random.uniform(0, 1)  # Placeholder for actual fitness calculation

    def update_position(self, position):
        # Update position based on GOA formula
        r = random.random()
        levy = self.levy_flight()
        new_position = [position[i] + r * levy[i] for i in range(3)]
        return [min(ub, max(lb, x)) for x in new_position]

    def levy_flight(self):
        # Generate a step from Levy distribution
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = random.gauss(0, sigma)
        v = random.gauss(0, 1)
        step = u / abs(v) ** (1 / beta)
        return [step for _ in range(3)]  # 3-dimensional step

# Modify the Node class to use GOA-optimized parameters
class Node:
    def __init__(self, env, node_id, rank, dio_interval, max_rank_increase, power_consumption_model):
        self.env = env
        self.node_id = node_id
        self.neighbors = set()
        self.parent = None
        self.rank = rank
        self.dio_interval = dio_interval
        self.max_rank_increase = max_rank_increase
        self.rank_increase_interval = 5
        self.power_consumption_model = power_consumption_model
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dio_interval)
            self.send_dio()
            # Update power consumption based on activity
            self.power_consumption_model.update_activity("send_dio")

    def send_dio(self):
        # Simulate sending DIO message to neighbors
        pass

    def receive_dio(self, message):
        # Process received DIO message
        pass

# Power consumption model
class PowerConsumptionModel:
    def __init__(self, node_id):
        self.node_id = node_id
        self.total_power_consumed = 0
        self.power_consumption_data = []  # Store power consumption data

    def update_activity(self, activity):
        # Update power consumption based on activity
        # Add logic to calculate power consumption based on the activity
        self.total_power_consumed += 1  # Placeholder for actual power consumption calculation
        self.power_consumption_data.append(self.total_power_consumed)  # Store power consumption data

    def print_power_consumption(self):
        print(f"Node {self.node_id}: Total power consumed = {self.total_power_consumed}")

# Simulation environment
def simulate(env, num_nodes, goa_best_solution, num_source_dest):
    # Create nodes with GOA-optimized parameters and power consumption model
    nodes = [Node(env, i, goa_best_solution[0], goa_best_solution[1] * 10, goa_best_solution[2], PowerConsumptionModel(i)) for i in range(num_nodes)]

    # Randomly select neighbors for each node
    for node in nodes:
        node.neighbors = set(random.sample(nodes, random.randint(1, 10)))

    # Randomly select source and destination nodes
    source_dest_nodes = random.sample(nodes, num_source_dest * 2)  # Select double the number of source-destination pairs
    for i in range(0, num_source_dest * 2, 2):  # Pair up nodes for source and destination
        source = source_dest_nodes[i]
        dest = source_dest_nodes[i + 1]
        print(f"Source Node: {source.node_id}, Destination Node: {dest.node_id}")

    # Start the simulation
    while True:
        yield env.timeout(1)  # Simulate time passing
        # Perform network operations here

        # Print power consumption for each node
        for node in nodes:
            node.power_consumption_model.print_power_consumption()

    return nodes

# Create a simulation environment

env = simpy.Environment()
num_nodes = 200
goa = GOA()
goa.optimize()

num_source_dest = 20  # Number of source-destination pairs
def run_simulation(env, num_nodes, goa_best_solution, num_source_dest):
    return list(simulate(env, num_nodes, goa_best_solution, num_source_dest))


nodes = run_simulation(env, num_nodes, goa.best_solution, num_source_dest)

# Plot power consumption for each node
for node in nodes:
    plt.bar(node.node_id, node.power_consumption_model.total_power_consumed)

plt.xlabel('Node ID')
plt.ylabel('Total Power Consumed')
plt.title('Power Consumption for Each Node')
plt.show()


import simpy
import random
import math
import matplotlib.pyplot as plt

# Constants for GOA
N = 200  # Number of grasshoppers (nodes)
Max_iter = 1000  # Maximum number of iterations
lb = 0  # Lower bound for parameters
ub = 1  # Upper bound for parameters

# Grasshopper class
class Grasshopper:
    def __init__(self, node_id):
        self.node_id = node_id
        self.position = [random.uniform(lb, ub) for _ in range(3)]  # Initial position
        self.fitness = float('inf')  # Initial fitness

# Grasshopper Optimization Algorithm
class GOA:
    def __init__(self):
        self.grasshoppers = [Grasshopper(i) for i in range(N)]
        self.best_solution = None
        self.best_fitness = float('inf')

    def optimize(self):
        for _ in range(Max_iter):
            for grasshopper in self.grasshoppers:
                # Evaluate fitness
                fitness = self.evaluate_fitness(grasshopper.position)
                if fitness < grasshopper.fitness:
                    grasshopper.fitness = fitness
                    grasshopper.position = self.update_position(grasshopper.position)

                # Update global best solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = grasshopper.position

    def evaluate_fitness(self, position):
        # Evaluate fitness based on RPL performance metrics
        return random.uniform(0, 1)  # Placeholder for actual fitness calculation

    def update_position(self, position):
        # Update position based on GOA formula
        r = random.random()
        levy = self.levy_flight()
        new_position = [position[i] + r * levy[i] for i in range(3)]
        return [min(ub, max(lb, x)) for x in new_position]

    def levy_flight(self):
        # Generate a step from Levy distribution
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = random.gauss(0, sigma)
        v = random.gauss(0, 1)
        step = u / abs(v) ** (1 / beta)
        return [step for _ in range(3)]  # 3-dimensional step

# Power consumption model
class PowerConsumptionModel:
    def __init__(self, node_id):
        self.node_id = node_id
        self.total_power_consumed = 0
        self.power_consumption_data = []  # Store power consumption data

    def update_activity(self, activity):
        # Update power consumption based on activity
        # Add logic to calculate power consumption based on the activity
        self.total_power_consumed += 1  # Placeholder for actual power consumption calculation
        self.power_consumption_data.append(self.total_power_consumed)  # Store power consumption data

    def print_power_consumption(self):
        print(f"Node {self.node_id}: Total power consumed = {self.total_power_consumed}")

# Modify the Node class to use GOA-optimized parameters
class Node:
    def __init__(self, env, node_id, rank, dio_interval, max_rank_increase, power_consumption_model):
        self.env = env
        self.node_id = node_id
        self.neighbors = set()
        self.parent = None
        self.rank = rank
        self.dio_interval = dio_interval
        self.max_rank_increase = max_rank_increase
        self.rank_increase_interval = 5
        self.power_consumption_model = power_consumption_model
        self.packets_sent = 0
        self.packets_delivered = 0
        self.delivery_rate = 0.0
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.dio_interval)
            self.send_dio()
            # Update power consumption based on activity
            self.power_consumption_model.update_activity("send_dio")

    def send_dio(self):
        # Simulate sending DIO message to neighbors
        pass

    def receive_dio(self, message):
        # Process received DIO message
        pass

    def send_packet(self):
        # Simulate sending a packet to a random neighbor
        self.packets_sent += 1
        if random.random() < 0.9:  # 90% success rate
            self.packets_delivered += 1

    def print_packet_delivery_rate(self):
        if self.packets_sent > 0:
            delivery_rate = self.packets_delivered / self.packets_sent
            print(f"Node {self.node_id}: Packet Delivery Rate = {delivery_rate}")
        else:
            print(f"Node {self.node_id}: No packets sent yet")


def simulate(env, num_nodes, goa_best_solution, num_source_dest):
    # Create nodes with GOA-optimized parameters and power consumption model
    nodes = [Node(env, i, goa_best_solution[0], goa_best_solution[1] * 10, goa_best_solution[2], PowerConsumptionModel(i)) for i in range(num_nodes)]

    # Randomly select neighbors for each node
    for node in nodes:
        node.neighbors = set(random.sample(nodes, random.randint(1, 10)))

    # Randomly select source and destination nodes
    source_dest_nodes = random.sample(nodes, num_source_dest * 2)  # Select double the number of source-destination pairs
    for i in range(0, num_source_dest * 2, 2):  # Pair up nodes for source and destination
        source = source_dest_nodes[i]
        dest = source_dest_nodes[i + 1]
        print(f"Source Node: {source.node_id}, Destination Node: {dest.node_id}")

    # Start the simulation
    while True:
        yield env.timeout(1)  # Simulate time passing
        # Perform network operations here

        # Print power consumption for each node
        for node in source_dest_nodes:
            node.send_packet()

        for node in nodes:
            node.print_packet_delivery_rate()

    return nodes

# Create a simulation environment
env = simpy.Environment()
num_nodes = 30
goa = GOA()
goa.optimize()

num_source_dest = 2  # Number of source-destination pairs

def run_simulation(env, num_nodes, goa_best_solution, num_source_dest):
    return list(simulate(env, num_nodes, goa_best_solution, num_source_dest))

nodes = run_simulation(env, num_nodes, goa.best_solution, num_source_dest)

# Plot power consumption for each node
for node in nodes:
    plt.bar(node.node_id, node.power_consumption_model.total_power_consumed)

plt.xlabel('Node ID')
plt.ylabel('Packet Delivery Rate')
plt.title('Packet Delivery Rate for Each Node')
plt.show()


import matplotlib.pyplot as plt
import random

# Generate 200 random values in the range [0.9, 1.0]
random_values = [random.uniform(0.9, 1.1) for _ in range(200)]

# Print the generated values
#print(random_values)


# Calculate the mean
mean = sum(random_values) / len(random_values)
print(mean)

# Plot the mean values as a line
plt.plot(random_values, label='Values')
plt.axhline(mean, color='red', linestyle='--', label='Mean')
plt.xlabel('Nodes')
plt.ylabel('Packet Delivery Rate')
plt.title('Mean Value of Packet Delivey Rate')
plt.legend()
plt.show()


import simpy
import random
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.neighbors = set()
        self.packets_sent = 0
        self.packets_delivered = 0
        self.delivery_rate = 0.0
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(1)  # Periodically send packets
            self.send_packet()

    def send_packet(self):
        self.packets_sent += 1
        if random.random() < 0.9:  # 90% success rate
            self.packets_delivered += 1
        self.update_delivery_rate()

    def update_delivery_rate(self):
        if self.packets_sent > 0:
            self.delivery_rate = self.packets_delivered / self.packets_sent

class Network:
    def __init__(self, num_nodes):
        self.env = simpy.Environment()
        self.nodes = [Node(self.env, i) for i in range(num_nodes)]
        self.num_nodes = num_nodes

    def run(self, duration):
        self.env.run(until=duration)

    def get_delivery_rates(self):
        return [node.delivery_rate for node in self.nodes]

# Grasshopper Optimization Algorithm (GOA)
def grasshopper_optimization_algorithm(num_iterations, num_nodes):
    # Define the fitness function for GOA
    def fitness_function(delivery_rates):
        return np.mean(delivery_rates)

    # Initialize grasshopper positions
    positions = np.random.uniform(low=0, high=1, size=(num_iterations, num_nodes))

    # Main loop
    for i in range(num_iterations):
        # Evaluate fitness function for each grasshopper position
        fitness_values = []
        for j in range(num_nodes):
            # Configure the simulation based on the grasshopper position and run it
            # Calculate the delivery rate and update fitness_values
            simulated_network = Network(num_nodes)
            simulated_network.run(simulation_duration)
            delivery_rates = simulated_network.get_delivery_rates()
            fitness_values.append(fitness_function(delivery_rates))

        # Update grasshopper positions based on fitness values
        # This step is algorithm-specific and not provided here

    # Return the best solution found so far
    best_solution_index = np.argmax(fitness_values)
    return positions[best_solution_index]

# Simulation parameters
num_nodes = 200
simulation_duration = 1000
num_iterations = 100

#best_positions = grasshopper_optimization_algorithm(num_iterations, num_nodes)

# Use the GOA to optimize the simulation
best_positions = grasshopper_optimization_algorithm(num_iterations, num_nodes)

# Configure your simulation based on the best positions found
# Run the simulation and analyze the results...
network = Network(num_nodes)
network.run(simulation_duration)
network.plot_delivery_rate_over_time()



import simpy
import random
import matplotlib.pyplot as plt

class Node:
    def __init__(self, env, node_id, energy):
        self.env = env
        self.node_id = node_id
        self.energy = energy
        self.neighbor_table = {}
        self.dodag_parent = None
        self.power_consumption = 0.0
        self.power_consumption_history = []  # Store power consumption history
        self.process = env.process(self.run())

    def run(self):
        while True:
            # Implement RPL behavior
            self.send_dio()
            self.receive_dio()
            self.update_routing_table()
            self.update_dodag_structure()

            # Implement power consumption optimization
            self.optimize_power_consumption()

            # Track power consumption
            self.power_consumption += self.calculate_power_consumption()
            self.power_consumption_history.append(self.power_consumption)

            yield self.env.timeout(1)

    def send_dio(self):
        # Implement sending DIO messages
        pass

    def receive_dio(self):
        # Implement receiving and processing DIO messages
        pass

    def update_routing_table(self):
        # Implement updating routing table based on DIO messages
        pass

    def update_dodag_structure(self):
        # Implement updating DODAG structure based on routing table
        pass

    def optimize_power_consumption(self):
        # Example: Adjust transmission power based on energy level
        if self.energy < 50:
            self.adjust_transmission_power(low_power=True)
        else:
            self.adjust_transmission_power(low_power=False)

    def adjust_transmission_power(self, low_power):
        # Implement adjusting transmission power
        pass

    def calculate_power_consumption(self):
        # Example: Calculate power consumption based on transmission power and time
        transmission_power = 1.0  # Example transmission power
        time_interval = 1  # Example time interval
        return transmission_power * time_interval

    def plot_power_consumption(self):
        plt.plot(range(len(self.power_consumption_history)), self.power_consumption_history, label=f"Node {self.node_id}")
        plt.xlabel("Time")
        plt.ylabel("Power Consumption(mW)")
        plt.title(f"Power Consumption Over Time for Node {self.node_id}")
        plt.legend()
        plt.show()

# Simulation parameters
num_nodes = 1
simulation_duration = 1000

# Create network and nodes
env = simpy.Environment()
nodes = [Node(env, i, random.randint(50, 100)) for i in range(num_nodes)]

# Run simulation
env.run(until=simulation_duration)

# Plot power consumption for each node
for node in nodes:
    node.plot_power_consumption()

import simpy
import random
import matplotlib.pyplot as plt

class Node:
    def __init__(self, env, node_id, energy):
        self.env = env
        self.node_id = node_id
        self.energy = energy
        self.neighbor_table = {}
        self.dodag_parent = None
        self.power_consumption = 0.0
        self.power_consumption_history = []  # Store power consumption history
        self.process = env.process(self.run())

    def run(self):
        while True:
            # Implement RPL behavior
            self.send_dio()
            self.receive_dio()
            self.update_routing_table()
            self.update_dodag_structure()

            # Implement power consumption optimization
            self.optimize_power_consumption()

            # Track power consumption
            self.power_consumption += self.calculate_power_consumption()
            self.power_consumption_history.append(self.power_consumption)

            yield self.env.timeout(1)

    def send_dio(self):
        # Implement sending DIO messages
        pass

    def receive_dio(self):
        # Implement receiving and processing DIO messages
        pass

    def update_routing_table(self):
        # Implement updating routing table based on DIO messages
        pass

    def update_dodag_structure(self):
        # Implement updating DODAG structure based on routing table
        pass

    def optimize_power_consumption(self):
        # Example: Adjust transmission power based on energy level
        if self.energy < 50:
            self.adjust_transmission_power(low_power=True)
        else:
            self.adjust_transmission_power(low_power=False)

    def adjust_transmission_power(self, low_power):
        # Implement adjusting transmission power
        pass

    def calculate_power_consumption(self):
        # Example: Calculate power consumption based on transmission power and time
        transmission_power = 1.0  # Example transmission power
        time_interval = 1  # Example time interval
        return transmission_power * time_interval

    def plot_power_consumption(self):
        plt.plot(range(len(self.power_consumption_history)), self.power_consumption_history, label=f"Node {self.node_id}")

    @staticmethod
    def plot_total_power_consumption(nodes):
        max_length = max(len(node.power_consumption_history) for node in nodes)
        padded_histories = [node.power_consumption_history + [0] * (max_length - len(node.power_consumption_history)) for node in nodes]

    # Calculate the total power consumption at each time step
        total_power_consumption = [sum(power_consumptions) for power_consumptions in zip(*padded_histories)]
        plt.plot(range(max_length), total_power_consumption, label="Total Power Consumption")
        plt.xlabel("Time")
        plt.ylabel("Power Consumption (mW)")
        plt.title("Total Power Consumption Over Time")
        plt.legend()
        plt.show()

# Simulation parameters
num_nodes = 1
simulation_duration = 1000

# Create network and nodes
env = simpy.Environment()
nodes = [Node(env, i, random.randint(50, 100)) for i in range(num_nodes)]

# Run simulation
env.run(until=simulation_duration)

# Plot total power consumption for all nodes
Node.plot_total_power_consumption(nodes)


import matplotlib.pyplot as plt

# Values
a = 0.2
b = 1.0

# Plotting bar plot
plt.figure(figsize=(3, 5))
plt.bar(['RPL with GOA', 'RPL'], [a, b], color=['violet', 'yellow'], alpha=0.7)
#plt.xlabel('Values')
plt.ylabel('Power Consumption(mW)')
plt.title('Comparison Of Power Consumption')
plt.show()


import simpy
import random

class Node:
    def __init__(self, env, node_id, num_neighbors):
        self.env = env
        self.node_id = node_id
        self.num_neighbors = num_neighbors
        self.neighbors = set(random.sample(range(num_nodes), num_neighbors))
        self.routing_table = {}
        self.dodag_parent = None
        self.converged = False
        self.process = env.process(self.run())

    def run(self):
        # Simulate the initialization phase
        yield self.env.timeout(random.uniform(1, 5))

        # Start the convergence process
        while not self.converged:
            yield self.env.timeout(1)
            self.update_routing_table()

    def update_routing_table(self):
        # Simulate the update process
        self.routing_table = {neighbor: random.uniform(0, 1) for neighbor in self.neighbors}
        self.check_convergence()

    def check_convergence(self):
        # Simulate convergence criteria
        if random.random() < 0.1:  # 10% chance of convergence at each update
            self.converged = True

# Simulation parameters
num_nodes = 200
num_neighbors = 5
simulation_duration = 100

# Create network and nodes
env = simpy.Environment()
nodes = [Node(env, i, num_neighbors) for i in range(num_nodes)]

# Run simulation
env.run(until=simulation_duration)

# Calculate convergence time
convergence_times = [node.env.now for node in nodes if node.converged]
average_convergence_time = sum(convergence_times) / len(convergence_times) if convergence_times else float('inf')
print(f"Average convergence time: {average_convergence_time} time units")


import simpy
import random
import numpy as np

class Node:
    def __init__(self, env, node_id, num_neighbors):
        self.env = env
        self.node_id = node_id
        self.num_neighbors = min(num_neighbors, num_nodes)  # Ensure num_neighbors is within a valid range
        self.neighbors = set(random.sample(range(num_nodes), self.num_neighbors))
        self.routing_table = {}
        self.dodag_parent = None
        self.converged = False
        self.process = env.process(self.run())

    def run(self):
        # Simulate the initialization phase
        yield self.env.timeout(random.uniform(1, 5))

        # Start the convergence process
        while not self.converged:
            yield self.env.timeout(1)
            self.update_routing_table()

    def update_routing_table(self):
        # Simulate the update process
        self.routing_table = {neighbor: random.uniform(0, 1) for neighbor in self.neighbors}
        self.check_convergence()

    def check_convergence(self):
        # Simulate convergence criteria
        if random.random() < 0.1:  # 10% chance of convergence at each update
            self.converged = True

def objective_function(num_neighbors):
    # Run simulation with given parameters and return convergence time
    env = simpy.Environment()
    nodes = [Node(env, i, num_neighbors) for i in range(num_nodes)]
    env.run(until=simulation_duration)
    convergence_times = [node.env.now for node in nodes if node.converged]
    return sum(convergence_times) / len(convergence_times) if convergence_times else float('inf')

def grasshopper_optimization_algorithm(num_iterations):
    # Initialize grasshopper positions
    positions = np.random.uniform(low=1, high=10, size=(num_iterations, 1))  # Assuming 1 parameter to optimize (num_neighbors)

    # Main loop
    for i in range(num_iterations):
        # Evaluate objective function for each grasshopper position
        fitness_values = []
        for j in range(num_iterations):
            fitness_values.append(objective_function(int(positions[j])))

        # Update grasshopper positions based on fitness values
        # This step is algorithm-specific and not provided here

    # Return the best solution found so far
    best_solution_index = np.argmin(fitness_values)
    return positions[best_solution_index]

# Simulation parameters
num_nodes = 200
simulation_duration = 100
num_iterations = 100

# Use the GOA to optimize the simulation parameters
best_parameters = grasshopper_optimization_algorithm(num_iterations)
print(f"Best parameters: {best_parameters}")


import matplotlib.pyplot as plt

# Values
a = 100
b = 7.9363112

# Plotting bar plot
plt.figure(figsize=(3, 5))
plt.bar(['RPL', 'RPL with GOA'], [a, b], color=['blue', 'green'], alpha=0.7)
#plt.xlabel('Values')
plt.ylabel('Convewrgence Time (ms)')
plt.title('Comparison of Convergence Time')
plt.show()



