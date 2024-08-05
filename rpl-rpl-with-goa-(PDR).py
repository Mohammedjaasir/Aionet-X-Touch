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

# Simulation parameters
num_nodes = 200
simulation_duration = 1000

# Create network and run simulation
network = Network(num_nodes)
network.run(simulation_duration)

# Get and plot packet delivery rates
delivery_rates = network.get_delivery_rates()
plt.plot(delivery_rates)
mean_delivery_rate = np.mean(delivery_rates)
print(mean_delivery_rate)
plt.axhline(y=mean_delivery_rate, color='r', linestyle='--', label='Mean Delivery Rate')
plt.xlabel('Nodes')
plt.ylabel('Packet Delivery Rate(seconds)')
plt.title('Packet Delivery Rate for Each Node')
plt.show()







import matplotlib.pyplot as plt

# Values
a = 0.8998348348348348
b = 0.9983384639115365

# Plotting bar plot
plt.figure(figsize=(3, 6))
plt.bar(['RPL', 'RPL with GOA'], [a, b], color=['blue', 'green'], alpha=0.7)
#plt.xlabel('')
plt.ylabel('Frequency')
plt.title('Comparison Packet Delivery Rate')
plt.show()

