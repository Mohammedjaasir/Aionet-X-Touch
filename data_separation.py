import pandas as pd

# Read the Excel file into a DataFrame
data = pd.read_excel("intrusion.xlsx")

# Add columns for model results
data['Model'] = ''
data['Accuracy'] = 0.0
data['Precision'] = 0.0
data['F1-measure'] = 0.0
data['Recall'] = 0.0

# Assuming you have results for each model in separate variables (replace with actual values)
ann_accuracy = 0.85
ann_precision = 0.82
ann_f1_measure = 0.78
ann_recall = 0.75

knn_accuracy = 0.75
knn_precision = 0.72
knn_f1_measure = 0.68
knn_recall = 0.65

decision_tree_accuracy = 0.80
decision_tree_precision = 0.78
decision_tree_f1_measure = 0.72
decision_tree_recall = 0.70

# Populate the data for each model
data.loc[data.index[0], 'Model'] = 'ANN'
data.loc[data.index[0], 'Accuracy'] = ann_accuracy
data.loc[data.index[0], 'Precision'] = ann_precision
data.loc[data.index[0], 'F1-measure'] = ann_f1_measure
data.loc[data.index[0], 'Recall'] = ann_recall

data.loc[data.index[1], 'Model'] = 'KNN'
data.loc[data.index[1], 'Accuracy'] = knn_accuracy
data.loc[data.index[1], 'Precision'] = knn_precision
data.loc[data.index[1], 'F1-measure'] = knn_f1_measure
data.loc[data.index[1], 'Recall'] = knn_recall

data.loc[data.index[2], 'Model'] = 'Decision Tree'
data.loc[data.index[2], 'Accuracy'] = decision_tree_accuracy
data.loc[data.index[2], 'Precision'] = decision_tree_precision
data.loc[data.index[2], 'F1-measure'] = decision_tree_f1_measure
data.loc[data.index[2], 'Recall'] = decision_tree_recall

# Display the integrated dataset
print(data)
