import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from ecpy.curves import Curve, Point

# Define parameters
NUM_NODES = 20
SIMULATION_TIME = 100  # Simulation time in seconds
TRANSMISSION_RANGE = 50  # Transmission range of nodes in meters

data_df = pd.read_csv("BotNeTIoT-L01_label_NoDuplicates.csv")

node_positions = {'x': [], 'y': []}
node_info = {}

# Define the elliptic curve parameters
curve = Curve.get_curve('secp256k1')  # Example curve (Bitcoin curve)

def iot_node(env, node_id, num_nodes, optimized_positions):
    mac_address = f"00:16:3e:4c:{random.randint(0,255)}:{random.randint(0,255)}"
    ip_address = f"192.168.0.{node_id + 1}"

    rpl_table = {i: {'parent': None, 'rank': float('inf')} for i in range(num_nodes)}
    rpl_table[node_id]['rank'] = 0  # Set rank of self to 0

    while True:
        data_packet = data_df.sample(n=1).iloc[0]  # Randomly select a row from the DataFrame
        for neighbor_id in range(num_nodes):
            if neighbor_id != node_id:
                neighbor_rank = rpl_table[node_id]['rank'] + 1
                if neighbor_rank < rpl_table[neighbor_id]['rank']:
                    rpl_table[neighbor_id]['rank'] = neighbor_rank
                    rpl_table[neighbor_id]['parent'] = node_id

        next_hop = min(rpl_table, key=lambda x: rpl_table[x]['rank'] if rpl_table[x]['parent'] is not None else float('inf'))

        transmission_time = random.uniform(0.1, 1.0)
        yield env.timeout(transmission_time)
        
        server_distance = optimized_positions[node_id]
        ipv4_header = {
            'TTL': random.randint(1, 255),
            'Protocol': random.choice(['TCP', 'UDP', 'ICMP']),
            'Header Checksum': f"{random.randint(0, 255):X}",
            'Source Address': ip_address,
            'Destination Address': f"Server IP",
            'Padding': '...',
            'Identification': random.randint(0, 65535),
            'Fragment Offset': random.randint(0, 8191),
            'Total Length': random.randint(20, 1500),
            'Version': 4,
            'IHL': 5,
            'ToS': f"{random.randint(0, 255):X}",
            'GOA': optimized_positions[node_id]
        }

        # Generate ECC key pair
        private_key = random.randint(1, curve.order - 1)
        public_key = private_key * curve.generator

        # Include ECC-related information in the IPv4 header
        ipv4_header['ECC'] = {
            'Curve': curve.name,
            'Private Key': private_key,
            'Public Key': public_key
        }

        node_positions['x'].append(server_distance)
        node_positions['y'].append(node_id)
        node_info[(server_distance, node_id)] = (f"Node {node_id}", f"MAC: {mac_address}", f"IP: {ip_address}",
                                                 ipv4_header)
        interval = random.uniform(10, 20)
        yield env.timeout(interval)

        print(f"Node {node_id} - GOA Position: {optimized_positions[node_id]} - IPv4 Header: {ipv4_header}")

# Grasshopper Optimization Algorithm (GOA)
def grasshopper_optimization(population_size, max_iterations, num_nodes, transmission_range):
    positions = [random.uniform(0, transmission_range) for _ in range(num_nodes)]
    # Implement GOA algorithm here to optimize positions
    optimized_positions = positions  # Placeholder for actual optimization

    return optimized_positions

# Initialize GOA parameters
population_size = 50
max_iterations = 100

# Run GOA to optimize node positions
optimized_positions = grasshopper_optimization(population_size, max_iterations, NUM_NODES, TRANSMISSION_RANGE)

# Create the simulation environment
env = simpy.Environment()

# Create IoT nodes and add them to the simulation environment
for i in range(NUM_NODES):
    env.process(iot_node(env, i, NUM_NODES, optimized_positions))

# Run the simulation
env.run(until=SIMULATION_TIME)

# Display the node positions using mplcursors
plt.figure(figsize=(10, 6))
plt.scatter(node_positions['x'], node_positions['y'], marker='o', s=40)
plt.xlabel('Distance to Server (m)')
plt.ylabel('Node ID')
plt.title('Node Positions')

# Create labels for mplcursors
labels = []
for _, info in node_info.items():
    label = f"Node {info[0]}\nMAC: {info[1]}\nIP: {info[2]}\nIPv4 Header:\n"
    for key, value in info[3].items():
        if key == 'ECC':
            label += f"\nECC:\n"
            for ecc_key, ecc_value in value.items():
                label += f"{ecc_key}: {ecc_value}\n"
        else:
            label += f"{key}: {value}\n"
    labels.append(label)

# Connect mplcursors to display labels
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(labels[sel.target.index]))

plt.grid(False)
plt.show()
