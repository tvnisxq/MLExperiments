# Importing required dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step: 2 Prepare the data
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'score': [12, 25, 32, 40, 50 , 55, 65, 72, 80, 90]
}

df = pd.DataFrame(data)

# Step 3: Create a binary target: Pass or Fail
# Students with marks greater than/equal to 50 are Pass  
df['Pass'] = (df['score'] >= 50).astype(int)

X = df[['hours_studied']]   # Feature
y = df['Pass']              # Target(0 or 1)

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions on the Test set
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_mat)

# Step 8.1: Create a ConfusionMatrixDisplay object 
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)

# Step 8.2: Plot the confusion matrix
disp.plot()

# Step 8.3: Save the plot as an image file
# plt.savefig('confusionmatrix.png')

# Step 8.4: Show the plot
plt.show()

# Step 9: Create a classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", class_report)

# Step 10: Visualize decision boundary
X_sorted = np.linspace(0, 12, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_sorted)[:, 1]

# Step: 11 plot the decision boundary
plt.scatter(X, y, color='blue', label='Actual Pass/Fail')
plt.plot(X_sorted, y_prob, color='red', label='Predicted Probability (Pass)')
plt.xlabel("Hours of study")
plt.ylabel("Probability of passing")
plt.title("Logistic Regression: Pass V/s Fail")
plt.legend()
plt.show()

