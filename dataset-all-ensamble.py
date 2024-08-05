import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.optimizers,keras.losses
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.impute import SimpleImputer


# Load the CSV file into a pandas DataFrame
data=pd.read_csv("BoTNeTIoT-L01-v2_2.csv")
print(data.head())
#print(data.columns)
#print(data.describe())
#print(data.isna().sum())
#print(data.info())
#print(len(data.columns))

# Load your dataset
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Labels (last column)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on X_train and transform X_train
X_train_imputed = imputer.fit_transform(X_train)
# Create a decision tree classifier
dt_clf = DecisionTreeClassifier()

# Train the decision tree classifier on the training data
dt_clf.fit(X_train, y_train)

# Make predictions on the test data using the decision tree classifier
dt_y_pred = dt_clf.predict(X_test)

# Calculate the accuracy of the decision tree classifier
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f"Decision Tree Accuracy: {dt_accuracy}")

# Create a KNN classifier
knn_clf = KNeighborsClassifier()

# Train the KNN classifier on the training data
knn_clf.fit(X_train, y_train)

# Make predictions on the test data using the KNN classifier
knn_y_pred = knn_clf.predict(X_test)

# Calculate the accuracy of the KNN classifier
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print(f"KNN Accuracy: {knn_accuracy}")

# Create an ANN classifier
ann_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Train the ANN classifier on the training data
ann_clf.fit(X_train, y_train)

# Make predictions on the test data using the ANN classifier
ann_y_pred = ann_clf.predict(X_test)

# Calculate the accuracy of the ANN classifier
ann_accuracy = accuracy_score(y_test, ann_y_pred)
print(f"ANN Accuracy: {ann_accuracy}")

