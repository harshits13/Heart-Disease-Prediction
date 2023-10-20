import numpy as np


class Logistic_Regression:

    # declaring learning rate & number of iterations (Hyper-parameters)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # fit function to train the model with dataset
    def fit(self, X, Y):
        # number of data points in the dataset (number of rows)  -->  m
        # number of input features in the dataset (number of columns)  --> n
        self.m, self.n = X.shape

        # initiating weight & bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # implementing Gradient Descent for Optimization

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        # y_output formula (sigmoid function)

        y_output = 1 / (1 + np.exp(- (self.X.dot(self.w) + self.b)))

        # derivatives

        dw = (1 / self.m) * np.dot(self.X.T, (y_output - self.Y))

        db = (1 / self.m) * np.sum(y_output - self.Y)

        # updating the weights & bias using gradient descent

        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db

    # Sigmoid Equation & Decision Boundary

    def predict(self, X):
        y_predicted = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        y_predicted = np.where(y_predicted > 0.5, 1, 0)
        return y_predicted
