import matplotlib.pyplot as plt

# Values
values = [0.9, 1.0, 1.0, 1.0]
labels = ['ECC', 'DES', 'AES', 'RSA']

# Create bar plot
plt.bar(labels, values, color=['skyblue','red','orange','yellow','darkblue'])

# Add title and labels
plt.title('Average Energy Consumptioin For Each Packet')
#plt.xlabel('Variables')
plt.ylabel('Power Consumption(ms)')

# Show plot
plt.show()



import matplotlib.pyplot as plt

# Values
values = [0.100000, 0.100405, 0.050219, 0.200606]
labels = ['ECC', 'DES', 'AES', 'RSA']

# Create bar plot
plt.bar(labels, values, color=['skyblue','red','orange','yellow','darkblue'])

# Add title and labels
plt.title('Average Packet Delay ')
#plt.xlabel('Variables')
plt.ylabel('Delay Rate(ms)')

# Show plot
plt.show()


import matplotlib.pyplot as plt

# Values
values = [256, 56,128, 2048]
labels = ['ECC', 'DES', 'AES', 'RSA']

# Create bar plot
plt.bar(labels, values, color=['skyblue','red','orange','yellow','darkblue'])

# Add title and labels
plt.title('Average Security Strenght ')
#plt.xlabel('Variables')
plt.ylabel('Keysize(bits)')

# Show plot
plt.show()


import matplotlib.pyplot as plt

# Values
values = [5.0,100.0, 50.0, 200.0]
labels = ['ECC', 'DES', 'AES', 'RSA']

# Create bar plot
plt.bar(labels, values, color=['skyblue','red','orange','yellow','darkblue'])

# Add title and labels
plt.title('Average Latency ')
#plt.xlabel('Variables')
plt.ylabel('Time(ms)')

# Show plot
plt.show()



import matplotlib.pyplot as plt

# Values
values = [10.0, 10.0, 20.0,30.0]
labels = ['ECC', 'DES', 'AES', 'RSA']

# Create bar plot
plt.bar(labels, values, color=['skyblue','red','orange','yellow','darkblue'])

# Add title and labels
plt.title('Average Rsource Utilization ')
#plt.xlabel('Variables')
plt.ylabel('Units')

# Show plot
plt.show()



import matplotlib.pyplot as plt

# Data
algorithms = ['ECC', 'DES', 'AES', 'RSA']
energy_consumption = [0.9, 1.0, 1.0, 1.0]
packet_delay = [0.100, 0.100405, 0.050219, 0.200606]
security_strength = [256, 56, 128, 2048]
latency = [5, 100, 50, 200]
resource_utilization = [10, 10, 20, 30]

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Average Energy Consumption
axes[0, 0].bar(algorithms, energy_consumption, color='skyblue')
axes[0, 0].set_title('Average Energy Consumption')
axes[0, 0].set_ylabel('Energy Consumption')

# Average Packet Delay
axes[0, 1].bar(algorithms, packet_delay, color='lightgreen')
axes[0, 1].set_title('Average Packet Delay (s)')
axes[0, 1].set_ylabel('Packet Delay')

# Average Security Strength
axes[1, 0].bar(algorithms, security_strength, color='salmon')
axes[1, 0].set_title('Average Security Strength (bits)')
axes[1, 0].set_ylabel('Security Strength')

# Average Latency
axes[1, 1].bar(algorithms, latency, color='gold')
axes[1, 1].set_title('Average Latency (ms)')
axes[1, 1].set_ylabel('Latency')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

