'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Generating some random data for demonstration
np.random.seed(48)
X = np.random.randn(1000, 2)
y = np.random.choice([0, 1], size=1000)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)

# Training a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Getting predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate accuracy, precision, and recall for the overall test set
overall_accuracy = accuracy_score(y_test, y_pred)
overall_precision = precision_score(y_test, y_pred)
overall_recall = recall_score(y_test, y_pred)

# Define thresholds for confidence levels
thresholds = np.linspace(0.1, 0.9, 9)

# Initialize lists to store performance metrics for different confidence levels
accuracy_scores = []
precision_scores = []
recall_scores = []

# Calculate performance metrics for each confidence level
for threshold in thresholds:
    # Filter predictions based on confidence level
    threshold_indices = np.where(y_prob >= threshold)
    y_thresholded = y_pred[threshold_indices]
    y_test_thresholded = y_test[threshold_indices]
    
    # Calculate metrics for the subset
    accuracy = accuracy_score(y_test_thresholded, y_thresholded)
    precision = precision_score(y_test_thresholded, y_thresholded)
    recall = recall_score(y_test_thresholded, y_thresholded)
    
    # Append to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)

# Plot error analysis curve
plt.figure()
plt.plot(thresholds, accuracy_scores, label='Accuracy')
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.xlabel('Confidence Threshold')
plt.ylabel('Score')
plt.title('Error Analysis Curve (DT with KNN)')
plt.legend()
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Generating some random data for demonstration
np.random.seed(48)
X = np.random.randn(1000, 2)
y = np.random.choice([0, 1], size=1000)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)

# Training an ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)  # Example of a simple ANN with two hidden layers
model.fit(X_train, y_train)

# Getting predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate accuracy, precision, and recall for the overall test set
overall_accuracy = accuracy_score(y_test, y_pred)
overall_precision = precision_score(y_test, y_pred)
overall_recall = recall_score(y_test, y_pred)

# Define thresholds for confidence levels
thresholds = np.linspace(0.1, 0.9, 9)

# Initialize lists to store performance metrics for different confidence levels
accuracy_scores = []
precision_scores = []
recall_scores = []

# Calculate performance metrics for each confidence level
for threshold in thresholds:
    # Filter predictions based on confidence level
    threshold_indices = np.where(y_prob >= threshold)
    y_thresholded = y_pred[threshold_indices]
    y_test_thresholded = y_test[threshold_indices]
    
    # Calculate metrics for the subset
    accuracy = accuracy_score(y_test_thresholded, y_thresholded)
    precision = precision_score(y_test_thresholded, y_thresholded)
    recall = recall_score(y_test_thresholded, y_thresholded)
    
    # Append to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)

# Plot error analysis curve
plt.figure()
plt.plot(thresholds, accuracy_scores, label='Accuracy')
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.xlabel('Confidence Threshold')
plt.ylabel('Score')
plt.title('Error Analysis Curve (ANN)')
plt.legend()
plt.show()'''


