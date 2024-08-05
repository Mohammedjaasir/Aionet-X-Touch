import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Given values of 'a' and 'b'
a = 0.99876
b = 0.98856

# Heights of the bars
heights = [a, b]

# Bar labels
labels = ['F-measure', 'Recall']

# Set figure size
plt.figure(figsize=(3, 5))  # Adjust width and height as needed

# Bar plot
plt.bar(labels, heights)

# Add title and labels
plt.title('Overall Performance')
plt.ylabel('Metrics')

# Show plot
plt.show()

