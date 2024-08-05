import simpy
import random
import pandas as pd
import time
import matplotlib.pyplot as plt
import mplcursors
from tabulate import tabulate
# Define parameters
NUM_NODES = 10
SIMULATION_TIME = 100  # Simulation time in seconds
TRANSMISSION_RANGE = 50  # Transmission range of nodes in meters
DIO_INTERVAL_MIN = 5  # Minimum interval for DIO messages in seconds
DIO_INTERVAL_MAX = 10  # Maximum interval for DIO messages in seconds

data_df = pd.read_csv("BotNeTIoT-L01_label_NoDuplicates.csv")

node_positions={'x':[],'y':[]}
node_info={}

# Define a function to represent an IoT device/node
def iot_node(env, node_id,num_nodes,optimized_positions):
    mac_address = f"00:16:3e:4c:{random.randint(0,255)}:{random.randint(0,255)}"
    ip_address = f"192.168.0.{node_id + 1}"

    rpl_table = {i: {'parent': None, 'rank': float('inf')} for i in range(num_nodes)}
    rpl_table[node_id]['rank'] = 0  # Set rank of self to 0

    while True:
        data_packet = data_df.sample(n=1).iloc[0]  # Randomly select a row from the DataFrame
        #print("randomly selected data packet from dataset")
        for neighbor_id in range(num_nodes):
            if neighbor_id != node_id:
                # Calculate rank of neighbor (for simplicity, just use hop count)
                neighbor_rank = rpl_table[node_id]['rank'] + 1
                # Update neighbor's rank and parent if better
                if neighbor_rank < rpl_table[neighbor_id]['rank']:
                    rpl_table[neighbor_id]['rank'] = neighbor_rank
                    rpl_table[neighbor_id]['parent'] = node_id

        # Select next hop based on RPL routing table (for simplicity, select node with lowest rank)
        next_hop = min(rpl_table, key=lambda x: rpl_table[x]['rank'] if rpl_table[x]['parent'] is not None else float('inf'))
        #print(data_packet)
        #print(f"Node {node_id}: Received data packet - {data_packet['MI_dir_L0.1_weight']}")  # Adjust column name here

        # Simulate data transmission time
        transmission_time = random.uniform(0.1, 1.0)  # Random transmission time between 0.1 to 1.0 seconds
        yield env.timeout(transmission_time)
        
        # Simulate data transmission to the server
        server_distance = random.uniform(0, TRANSMISSION_RANGE)  # Random distance to the server
        #print(f"Node {node_id}: Data transmitted to server at time {env.now}, Distance to server: {server_distance} meters")
        # Simulate IPv4 header information
        ipv4_header = {
            'TTL': random.randint(1, 255),
            'Protocol': random.choice(['TCP', 'UDP', 'ICMP']),
            'Header Checksum': f"{random.randint(0, 255):X}",
            'Source Address': f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            'Destination Address': f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            'Padding': '...',
            'Identification': random.randint(0, 65535),
            'Fragment Offset': random.randint(0, 8191),
            'Total Length': random.randint(20, 1500),
            'Version': 4,
            'IHL': 5,
            'ToS': f"{random.randint(0, 255):X}",
            'GOA': optimized_positions[node_id] # Add GOA optimization information here
            
        }
        node_positions['x'].append(server_distance)
        node_positions['y'].append(node_id)
        node_info[(server_distance, node_id)] = (f"Node {node_id}", f"MAC: {mac_address}", f"IP: {ip_address}",
                                                 ipv4_header)
        # Simulate time until next data transmission
        interval = random.uniform(10, 20)  # Random interval between 10 to 20 seconds
        yield env.timeout(interval)

import random
import time

def calculate_power_consumption(node_positions, transmission_range):
    power_consumption_per_meter = 0.1  # Adjust as needed
    idle_power_consumption = 1.0  # Adjust as needed

    total_power_consumption = 0.0
    for server_distance in node_positions.values():
        if server_distance <= transmission_range:
            transmission_power_consumption = server_distance * power_consumption_per_meter
        else:
            transmission_power_consumption = float('inf')  # Considered as infinite power consumption

        total_power_consumption += transmission_power_consumption

    return total_power_consumption + (idle_power_consumption * len(node_positions))

def calculate_convergence_time(node_positions):
    converged_nodes = {node_id: False for node_id in node_positions.keys()}
    start_time = time.time()

    while not all(converged_nodes.values()):
        for node_id, position in node_positions.items():
            pass  # Implement convergence criteria for each node

    convergence_time = time.time() - start_time
    return convergence_time

def calculate_packet_delivery_rate(node_positions, transmission_range, num_packets):
    successful_deliveries = 0

    for server_distance in node_positions.values():
        if server_distance <= transmission_range:
            successful_deliveries += 1

    packet_delivery_rate = successful_deliveries / num_packets
    return packet_delivery_rate

# Example usage
TRANSMISSION_RANGE = 50  # Example transmission range
data_df = pd.read_csv("BotNeTIoT-L01_label_NoDuplicates.csv")

node_positions = {node_id: random.uniform(0, TRANSMISSION_RANGE) for node_id in range(NUM_NODES)}
transmission_range = 50  # Example transmission range
power_consumption = calculate_power_consumption(node_positions, transmission_range)
print(f"Total power consumption: {power_consumption}")

convergence_time = calculate_convergence_time(node_positions)
print(f"Convergence time: {convergence_time} seconds")

num_packets =  data_df.sample(n=1).iloc[0]  # Example total number of packets sent
delivery_rate = calculate_packet_delivery_rate(node_positions, transmission_range, num_packets)
print(f"Packet Delivery Rate: {delivery_rate}")



def initialize_population(num_nodes, transmission_range):
    return {node_id: random.uniform(0, transmission_range) for node_id in range(num_nodes)}

def fitness_function(positions, transmission_range, num_packets):
    node_positions = {node_id: pos for node_id, pos in enumerate(positions)}
    power_consumption = calculate_power_consumption(node_positions, transmission_range)
    convergence_time = calculate_convergence_time(node_positions)
    packet_delivery_rate = calculate_packet_delivery_rate(node_positions, transmission_range, num_packets)
    fitness_value = some_weighting * power_consumption + another_weighting * convergence_time + final_weighting * packet_delivery_rate
    return fitness_value

# Grasshopper Optimization Algorithm (GOA)
def grasshopper_optimization(population_size, max_iterations, num_nodes, transmission_range, num_packets):
    # Initialization
    positions = [random.uniform(0, transmission_range) for _ in range(num_nodes)]
    fitness_values = [fitness_function(positions, transmission_range, num_packets) for _ in range(population_size)]
    global_best_position = positions[fitness_values.index(min(fitness_values))]
    global_best_fitness = min(fitness_values)

    # Main loop
    for _ in range(max_iterations):
        # Update positions
        for i in range(population_size):
            new_positions = []
            for j in range(num_nodes):
                # Calculate new position for each node
                # Update node positions based on the GOA equations
                new_pos = ...  # Update node position using GOA equations
                new_positions.append(new_pos)

            # Calculate fitness of new positions
            new_fitness = fitness_function(new_positions, transmission_range, num_packets)

            # Update position if new fitness is better
            if new_fitness < fitness_values[i]:
                positions[i] = new_positions
                fitness_values[i] = new_fitness

        # Update global best position and fitness
        if min(fitness_values) < global_best_fitness:
            global_best_position = positions[fitness_values.index(min(fitness_values))]
            global_best_fitness = min(fitness_values)

    return global_best_position

# Initialize GOA parameters
population_size = 50
max_iterations = 100
some_weighting = 20
another_weighting = 20
final_weighting = 20

# Run GOA to optimize node positions
optimized_positions = grasshopper_optimization(population_size, max_iterations, NUM_NODES, TRANSMISSION_RANGE, num_packets)
# Create the simulation environment
env = simpy.Environment()

# Create IoT nodes and add them to the simulation environment
for i in range(NUM_NODES):
    env.process(iot_node(env, i,NUM_NODES))

# Run the simulation
env.run(until=SIMULATION_TIME)



"""plt.figure(figsize=(10, 6))
plt.scatter(node_positions['x'], node_positions['y'], marker='o', s=40)
plt.xlabel('Distance to Server (m)')
plt.ylabel('Node ID')
plt.title('Node Positions')"""

plt.figure(figsize=(10, 6))
plt.scatter(optimized_positions, range(NUM_NODES), marker='o', s=40)
plt.xlabel('Distance to Server (m)')
plt.ylabel('Node ID')
plt.title('Optimized Node Positions')


# Create labels for mplcursors
# Create labels for mplcursors
# Create labels for mplcursors
"""labels = [f"Node: {info[0]}\nMAC: {info[1]}\nIP: {info[2]}\nIPv4 Header:\n" +
          '\n'.join([f"{key}: {value}" for key, value in info[3].items() if value != 'N/A']) for _, info in node_info.items()]"""
labels = [f"Node {node_id}: Distance to Server = {pos} meters" for node_id, pos in enumerate(optimized_positions)]
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))


# Annotate nodes with MAC and IP addresses using mplcursors
#mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

mplcursors.cursor(hover=True)
plt.grid(False)
plt.show()

