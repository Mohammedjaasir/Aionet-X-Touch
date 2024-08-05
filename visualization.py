import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic data for binary classification
np.random.seed(0)
X = np.random.randn(20, 2)  # 20 examples with 2 features
y = np.random.choice([0, 1], size=20)  # Binary labels: 0 or 1

# Define and train a simple neural network
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0)

# Create a meshgrid to visualize decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z > 0.5).astype(int)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='+', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='o', label='1')
plt.title('Binary Classification with Artificial Neural Network')
plt.xlabel('Independent')
plt.ylabel('Dependent')
plt.legend()
plt.show()
