import matplotlib.pyplot as plt

# Data
algorithms = ['ECC', 'DES', 'AES', 'RSA']
energy_consumption = [0.9, 1.0, 1.0, 1.0]
packet_delay = [0.100, 0.100405, 0.050219, 0.100606]
security_strength = [256, 56, 128, 200]
latency = [5, 100, 50, 50]
resource_utilization = [30, 10, 20, 30]

# Calculate overall performance score for each algorithm
overall_performance = [sum(scores) for scores in zip(energy_consumption, packet_delay, security_strength, latency, resource_utilization)]

# Plotting
plt.figure(figsize=(5, 3))

# Overall Performance
plt.bar(algorithms, overall_performance, color='lightblue')
plt.title('Overall Performance of Algorithms')
#plt.xlabel('Algorithms')
plt.ylabel('Overall Performance Score')

# Show plot
plt.show()

