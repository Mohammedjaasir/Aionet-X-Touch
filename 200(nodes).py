import simpy
import random
import ipaddress
import matplotlib.pyplot as plt
import numpy as np
import time
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
import csv
import mplcursors
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
        # Define the objective function based on the provided metrics
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
        print(f"Best fitness value: {self.best_fitness}")
        return self.best_fitness

class IoTNode:
    """Class representing an IoT node."""

    def __init__(self, env, node_id, position, ipv6_address, ecc_key, payload_size):
        """
        Initialize an IoT node.

        Parameters:
            env (simpy.Environment): Simulation environment.
            node_id (int): Unique identifier for the node.
            position (tuple): Position of the node in (x, y) coordinates.
            ipv6_address (ipaddress.IPv6Address): IPv6 address of the node.
        """
        self.env = env
        self.node_id = node_id
        self.position = position
        self.payload_size = payload_size
        self.payload = "varible"
        self.bits_in_ecc = len(bin(int(ecc_key, 16))[2:])
        private_key = ec.derive_private_key(int(ecc_key, 16), ec.SECP256R1(), default_backend())
        public_key = private_key.public_key()

        # Serialize the keys to PEM format
        self.private_key_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                                         format=serialization.PrivateFormat.TraditionalOpenSSL,
                                                         encryption_algorithm=serialization.NoEncryption())
        self.public_key_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                                      format=serialization.PublicFormat.SubjectPublicKeyInfo)

        # Calculate the number of bits in the ECC key
        self.bits_in_ecc = len(bin(int(ecc_key, 16))[2:])
        self.ipv6_address = ipv6_address
        self.ecc_key = ecc_key 
        self.rpl_format = {
            "RPL Control Header": (1, 16),  
            "Source Address": (ipv6_address, 128),  
            "Destination Address": (ipaddress.IPv6Address('ff02::1a'), 128),  
            "RPL Instance ID": (random.randint(0, 255), 8),  
            "DODAG ID": (ipaddress.IPv6Address('2001:0db8:85a3:0000:0000:8a2e:0370:7334'), 128),  
            "Upper-layer Protocol": (17, 8),  
            "Extension Header Field (ECC Public Key)": (self.public_key_pem.decode(), self.bits_in_ecc),  # Public key
            "Extension Header Field (ECC Private Key)": (self.private_key_pem.decode(), self.bits_in_ecc),
            "Payload": (self.payload, self.payload_size)  # Private key
        }

class AttackerNode(IoTNode):
    """Class representing an attacker node."""
    
    def __init__(self, env, node_id, position, ipv6_address, ecc_key, payload_size, attack_type):
        super().__init__(env, node_id, position, ipv6_address, ecc_key, payload_size)
        self.attack_type = attack_type   
    
    def generate_payload(self):
        """Generate random payload data."""
        while True:
            if self.attack_type == "DoS":
                # Perform Denial of Service attack by not generating any payload
                yield self.env.timeout(1e9)  # Sleep indefinitely
            else:
                # Generate payload as usual for other attack types or normal nodes
                yield from super().generate_payload()

class Packet:
    """Class representing a packet."""
    
    def __init__(self, frame_number, frame_time, source_ip, destination_ip, udp_length, attack_type, label):
        self.frame_number = frame_number
        self.frame_time = frame_time
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.udp_length = udp_length
        self.attack_type = attack_type
        self.label = label

class IoTNetwork:
    """Class representing an IoT network with Grasshopper optimization."""

    def __init__(self, num_nodes, sim_duration, num_dimensions, num_agents, max_iter, payload_size):
        self.num_nodes = num_nodes
        self.sim_duration = sim_duration
        self.num_dimensions = num_dimensions
        self.num_agents = num_agents
        self.payload_size = payload_size
        self.max_iter = max_iter
        self.env = simpy.Environment()
        self.nodes = []
        self.packet_list = [] 
        
        self.frame_numbers = []
        self.frame_times = []
        self.source_ips = []
        self.destination_ips = []
        self.udp_lengths = []
        self.attack_types = []
        self.labels = []
 # List to store generated packets

    def generate_packets(self):
        # Generate packets for a fixed duration
        for _ in range(self.sim_duration):
            for node in self.nodes:
                # Check if the node is an attacker node
                if isinstance(node, AttackerNode):
                    # Annotate packets from attacker nodes with their attack type and label
                    attack_type = node.attack_type
                    label = "Attack"  # Set label to "Attack" for packets from attacker nodes
                else:
                    attack_type = "None"
                    label = "Normal"  # Set label to "Normal" for packets from normal nodes

                # Generate packet information
                frame_number = random.randint(1, 1000)
                frame_time = self.env.now
                source_ip = node.ipv6_address
                destination_ip = "ff02::1a"
                udp_length = random.randint(1, 100)

                self.frame_numbers.append(frame_number)
                self.frame_times.append(frame_time)
                self.source_ips.append(str(source_ip))
                self.destination_ips.append(destination_ip)
                self.udp_lengths.append(udp_length)
                self.attack_types.append(attack_type)
                self.labels.append(label)
                
                # Create a Packet object and add it to the packet list
                packet = Packet(frame_number, frame_time, source_ip, destination_ip, udp_length, attack_type, label)
                self.packet_list.append(packet)

                print(f"Node ID: {node.node_id}, Frame Number: {frame_number}, Frame Time: {frame_time}, Source IP: {source_ip}, Destination IP: {destination_ip}, UDP Length: {udp_length}, Attack Type: {attack_type}, Label: {label}")

            # Wait for a short duration before generating the next set of packets
            yield self.env.timeout(1)

    def visualize_packets(self):
        # Plot packet attributes
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(self.frame_numbers, self.frame_times, c=self.labels, cmap='viridis')

        node_info = []
        for node, packet in zip(self.nodes, self.packet_list):
            info = f"Node ID: {node.node_id}\nIPv6 Header: {node.ipv6_address}\nECC Key: {node.ecc_key}\n"
            info += f"Frame Number: {packet.frame_number}\nFrame Time: {packet.frame_time}\n"
            info += f"Source IP: {packet.source_ip}\nDestination IP: {packet.destination_ip}\n"
            info += f"UDP Length: {packet.udp_length}\nAttack Type: {packet.attack_type}\nLabel: {packet.label}"
            node_info.append(info)

        # Use mplcursors to display node and packet info on cursor hover
        mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(node_info[sel.target.index]))

    def run_simulation(self):
        # Initialize Grasshopper optimization algorithm
        goa = Grasshopper(self.num_dimensions, self.num_agents, self.max_iter)
        best_fitness = goa.optimize()

        # Create IoT nodes with IPv6 addresses
        for node_id in range(self.num_nodes):
            # Check if this node should be an attacker
            if random.random() < 0.1:  # 10% chance of being an attacker
                attack_type = random.choice(["MITM", "selective_forwarding", "DoS", "rank_manipulation"])
                node = AttackerNode(self.env, node_id, (random.uniform(0, 100), random.uniform(0, 100)),
                                    ipaddress.IPv6Address(f"2001:db8::{node_id}"),
                                    str(random.randint(0, 2**128 - 1)), self.payload_size, attack_type)
            else:
                node = IoTNode(self.env, node_id, (random.uniform(0, 100), random.uniform(0, 100)),
                               ipaddress.IPv6Address(f"2001:db8::{node_id}"),
                               str(random.randint(0, 2**128 - 1)), self.payload_size)

            self.nodes.append(node)

        # Start packet generation process
        self.env.process(self.generate_packets())

        # Run the simulation
        self.env.run(until=self.sim_duration)

        self.visualize_packets()


# Example usage:
num_nodes = 200
sim_duration = 100
num_dimensions = 2
num_agents = 20
max_iter = 100
payload_size = 128

network = IoTNetwork(num_nodes, sim_duration, num_dimensions, num_agents, max_iter, payload_size)
network.run_simulation()
plt.show()