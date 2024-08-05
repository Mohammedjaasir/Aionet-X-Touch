from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dataset
data = pd.read_csv("intrusion_1.csv")
print(data.columns)

# Perform any necessary preprocessing, such as encoding categorical variables

# Split the dataset into features (X) and labels (y)
X = pd.get_dummies(data.drop(columns=['Node ID']))
y = data['Attack']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=48)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
y_pred = model.predict_classes()
y_pred = y_pred.flatten()  # Flatten predictions to match y_test shape



# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Print metrics
print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)

bagging_model = BaggingClassifier(base_estimator=model, n_estimators=10, random_state=48)
bagging_model.fit(X_train_scaled, y_train)
bagging_y_pred = bagging_model.predict(X_test_scaled)

# Calculate bagging metrics
bagging_accuracy = accuracy_score(y_test, bagging_y_pred)
bagging_precision = precision_score(y_test, bagging_y_pred)
bagging_recall = recall_score(y_test, bagging_y_pred)
bagging_f1 = f1_score(y_test, bagging_y_pred)

# Boosting ensemble
boosting_model = AdaBoostClassifier(base_estimator=model, n_estimators=10, random_state=48)
boosting_model.fit(X_train_scaled, y_train)
boosting_y_pred = boosting_model.predict(X_test_scaled)

# Calculate boosting metrics
boosting_accuracy = accuracy_score(y_test, boosting_y_pred)
boosting_precision = precision_score(y_test, boosting_y_pred)
boosting_recall = recall_score(y_test, boosting_y_pred)
boosting_f1 = f1_score(y_test, boosting_y_pred)

stacking_estimators = [
    ('decision_tree', DecisionTreeClassifier()),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=48))
]
stacking_model = StackingClassifier(estimators=stacking_estimators, final_estimator=model)
stacking_model.fit(X_train_scaled, y_train)
stacking_y_pred = stacking_model.predict(X_test_scaled)

# Calculate stacking metrics
stacking_accuracy = accuracy_score(y_test, stacking_y_pred)
stacking_precision = precision_score(y_test, stacking_y_pred)
stacking_recall = recall_score(y_test, stacking_y_pred)
stacking_f1 = f1_score(y_test, stacking_y_pred)

# Voting ensemble
voting_estimators = [
    ('decision_tree', DecisionTreeClassifier()),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=48))
]
voting_model = VotingClassifier(estimators=voting_estimators)
voting_model.fit(X_train_scaled, y_train)
voting_y_pred = voting_model.predict(X_test_scaled)

# Calculate voting metrics
voting_accuracy = accuracy_score(y_test, voting_y_pred)
voting_precision = precision_score(y_test, voting_y_pred)
voting_recall = recall_score(y_test, voting_y_pred)
voting_f1 = f1_score(y_test, voting_y_pred)

# Random Subspace Method
random_subspace_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                          n_estimators=10, max_features=0.5, random_state=48)
random_subspace_model.fit(X_train_scaled, y_train)
random_subspace_y_pred = random_subspace_model.predict(X_test_scaled)

# Calculate random subspace metrics
random_subspace_accuracy = accuracy_score(y_test, random_subspace_y_pred)
random_subspace_precision = precision_score(y_test, random_subspace_y_pred)
random_subspace_recall = recall_score(y_test, random_subspace_y_pred)
random_subspace_f1 = f1_score(y_test, random_subspace_y_pred)

# Random Patches Method
random_patches_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                         n_estimators=10, max_samples=0.5, max_features=0.5, random_state=48)
random_patches_model.fit(X_train_scaled, y_train)
random_patches_y_pred = random_patches_model.predict(X_test_scaled)

# Calculate random patches metrics
random_patches_accuracy = accuracy_score(y_test, random_patches_y_pred)
random_patches_precision = precision_score(y_test, random_patches_y_pred)
random_patches_recall = recall_score(y_test, random_patches_y_pred)
random_patches_f1 = f1_score(y_test, random_patches_y_pred)

# Print ensemble metrics
print("Stacking Metrics:")
print("Accuracy:", stacking_accuracy)
print("Precision:", stacking_precision)
print("Recall:", stacking_recall)
print("F1 Score:", stacking_f1)

print("\nVoting Metrics:")
print("Accuracy:", voting_accuracy)
print("Precision:", voting_precision)
print("Recall:", voting_recall)
print("F1 Score:", voting_f1)

print("\nRandom Subspace Method Metrics:")
print("Accuracy:", random_subspace_accuracy)
print("Precision:", random_subspace_precision)
print("Recall:", random_subspace_recall)
print("F1 Score:", random_subspace_f1)

print("\nRandom Patches Method Metrics:")
print("Accuracy:", random_patches_accuracy)
print("Precision:", random_patches_precision)
print("Recall:", random_patches_recall)
print("F1 Score:", random_patches_f1)

# Print ensemble metrics
print("Bagging Metrics:")
print("Accuracy:", bagging_accuracy)
print("Precision:", bagging_precision)
print("Recall:", bagging_recall)
print("F1 Score:", bagging_f1)

print("\nBoosting Metrics:")
print("Accuracy:", boosting_accuracy)
print("Precision:", boosting_precision)
print("Recall:", boosting_recall)
print("F1 Score:", boosting_f1)
# Stacking ensemble'''
