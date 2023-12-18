import numpy as np

# Create a logistic regression class

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Fit the training data
    def fit(self, X, y):
        # Init parameters
        n_samples, n_features = X.shape   # Get the number of samples and features
        self.weights = np.zeros(n_features)     # Initialize the weights to 0
        self.bias = 0      # Initialize the bias to 0
        # Gradient descent
        for _ in range(self.n_iters):   # Loop through the number of iterations
           linear_model = np.dot(X, self.weights) + self.bias    # Calculate the linear model
           y_predicted = self._sigmoid(linear_model)   # Calculate the predicted y
           # Update the weights and bias
           dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))   # Calculate the derivative of the weights
           db = (1 / n_samples) * np.sum(y_predicted - y)    # Calculate the derivative of the bias
           self.weights -= self.lr * dw
           self.bias -= self.lr * db

    # Predict the test data
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias    # Calculate the linear model
        y_predicted = self._sigmoid(linear_model)     # Calculate the predicted y
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]   # Convert the predicted y to 0 or 1
        return y_predicted_cls
    
    # Private function to calculate the sigmoid function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))   # Sigmoid function