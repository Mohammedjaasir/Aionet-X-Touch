import matplotlib.pyplot as plt

# Values for different plots
values_delivery_ratio = [0.629, 0.950]
values_loss_ratio = [0.371, 0.010]
values_energy_consumption = [91, 5]
network_life_times = [1000, 800]
attacks_detected = [91, 5]
labels = ['RPL', 'RPL with ECC']

# Set the figure size and create subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 18))

# Plot 1: Packet Delivery Ratio
axs[0, 0].plot(labels, values_delivery_ratio, marker='o', color='b', linestyle='-', linewidth=2)
axs[0, 0].set_title('Packet Delivery Ratio')
axs[0, 0].set_ylabel('Delivery Ratio')

# Plot 2: Packet Loss Ratio
axs[0, 1].bar(labels, values_loss_ratio, color=['b', 'g'], edgecolor='black')
for i, value in enumerate(values_loss_ratio):
    axs[0, 1].text(i, value + 0.005, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
axs[0, 1].set_title('Packet Loss Ratio')
axs[0, 1].set_ylabel('Loss Ratio')

# Plot 3: Energy Consumption Per Node and Attack
axs[1, 0].boxplot([values_energy_consumption, attacks_detected], labels=['Energy Consumption', 'attacks'])
axs[1, 0].set_title('Energy Consumption per Node')
axs[1, 0].set_ylabel('Value')

# Plot 4: Network Life Time Comparison
axs[1, 1].bar(labels, network_life_times, color=['red', 'yellow'], edgecolor='black')
for i, value in enumerate(network_life_times):
    axs[1, 1].text(i, value + 20, f'{value}', ha='center', va='bottom', fontsize=12)
axs[1, 1].set_title('Network Life Time Comparison')
axs[1, 1].set_ylabel('Time (seconds)')

# Plot 5: Number of Attacks Detected
axs[2, 0].scatter(labels, attacks_detected, color=['red', 'green'], marker='o', s=100)
for i, value in enumerate(attacks_detected):
    axs[2, 0].text(i, value + 5, f'{value}', ha='center', va='bottom', fontsize=10)
axs[2, 0].set_title('Number of Attacks Detected')
axs[2, 0].set_ylabel('Attacks Detected')

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()

