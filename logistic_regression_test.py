import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from logistic_regression import LogisticRegression

# Load the dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target    # Get the data and target values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)   # Split the data into training and testing sets

# Create a logistic regression classifier
regressor = LogisticRegression(lr=0.0001, n_iters=1000)   # Create a logistic regression classifier
regressor.fit(X_train, y_train)     # Fit the training data
predictions = regressor.predict(X_test)    # Predict the test data

# Calculate the accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)   # Calculate the accuracy
    return accuracy

print("Logistic Regression classification accuracy:", accuracy(y_test, predictions))   # Print the accuracy
