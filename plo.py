import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Simulated data (replace with your actual data)
y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true))
plt.yticks(tick_marks, np.unique(y_true))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
