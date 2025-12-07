import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier 
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
data = {
    "Age": [25, 32, 47, 52, 28, 36, 49, 31, 45, 29],
    "Income": [45000, 60000, 30000, 80000, 40000, 52000, 65000, 58000, 72000, 41000],
    "LoanAmount": [500, 7000, 9000, 10000, 6500, 8200, 6000, 7500, 8800, 7200],
    "CreditScore": [680, 710, 600, 720, 590, 650, 705, 690, 730, 640],
    "Approved": [1, 1, 0, 1, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Split the data into features and target
X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
y = df['Approved'] 

# Initialize XGBoost model
model = XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=50, eval_metric='logloss')

# Cross Validation(k=5)
cv = 5
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())

# Make predictions using cross-validation
y_pred = cross_val_predict(model, X, y, cv=cv)


# Create a confusion matrix
conf_mat = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:\n", conf_mat)

# Create a ConfusionMatrixDisplayObject
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)

# Plot the confusion matrix
disp.plot()
disp.ax_.set_xlabel('Features: User Info')
disp.ax_.set_ylabel('Target: Approval')
disp.ax_.set_title('XGBoost: Y/N')
plt.show()