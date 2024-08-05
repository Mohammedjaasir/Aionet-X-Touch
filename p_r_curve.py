'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Calculating precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Calculating Area Under the Curve (AUC)
pr_auc = auc(recall, precision)

# Plotting precision-recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generating some random data for demonstration
np.random.seed(48)
X = np.random.randn(1000, 2)
y = np.random.choice([0, 1], size=1000)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the architecture of the ANN
model = Sequential([
    Dense(4, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Getting predicted probabilities
y_prob = model.predict(X_test).ravel()

# Calculating precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Calculating Area Under the Curve (AUC)
pr_auc = auc(recall, precision)

# Plotting precision-recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()'''

