'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Generating some random data for demonstration
np.random.seed(48)
X = np.random.randn(1000, 2)
y = np.random.choice([0, 1], size=1000)

# Define logistic regression model
model = LogisticRegression()

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 different training set sizes
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'  # Use accuracy as the evaluation metric
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
plt.xlabel("Training Set")
plt.ylabel("Score")
plt.title("Learning Curve (DT With KNN)")
plt.legend(loc="best")
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier

# Generating some random data for demonstration
np.random.seed(48)
X = np.random.randn(1000, 2)
y = np.random.choice([0, 1], size=1000)

# Define ANN model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10)  # Example of a simple ANN with two hidden layers

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 different training set sizes
    cv=5,  # 5-fold cross-validation
    scoring='accuracy'  # Use accuracy as the evaluation metric
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")
plt.xlabel("Training Set")
plt.ylabel("Score")
plt.title("Learning Curve (ANN)")
plt.legend(loc="best")
plt.show()'''


