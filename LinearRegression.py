import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

class LinearRegression:
    def __init__(self, iteration=100000, alpha=0.001):
        # Array with a and b from a*x+b
        self.theta = [0 ,0]
        # Amount of learning loop
        self.iteration = iteration
        # learning rate : step size for each iteration
        self.alpha = alpha
        self.cost = []
    
    def rescale_feature(self, feature):
        """
        Reshape the feature to get his mean at 0 and his range between -1 and 1
        """
        #Get average and range
        feature_avg = np.sum(feature) / len(feature)
        feature_range = np.max(feature) - np.min(feature)
        #Center at 0
        feature = feature - feature_avg
        #get range to [-1 : 1]
        feature = feature / feature_range
        return feature

    def unscale_theta(self, x):
        """
        "Unscales" the theta found.
        If we don't unscale it, predictions will be kinda wrong cause scaled down.
        """

        x_avg = sum(x) / len(x)
        x_range = max(x) - min(x)
        self.theta[0] = self.theta[0] - self.theta[1] * x_avg / x_range
        self.theta[1] = self.theta[1] / (max(x) - min(x))

    def train(self, raw_x, y):
        x = self.rescale_feature(raw_x)
        b, m = self.theta # in therory [0,0]
        n = len(x)
        for i in range(self.iteration):
            y_predicted = m * x + b
            cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
            md = -(2/n)*sum(x*(y-y_predicted))
            bd = -(2/n)*sum(y-y_predicted)
            m = m - self.alpha * md
            b = b - self.alpha * bd
            self.cost.append(cost)
        self.theta = [b, m]
        self.unscale_theta(raw_x)
        print("m {}, b {}, cost {}, iteration {}".format(self.theta[1], self.theta[0], cost, i))
    

    def predict(self, x):
        return self.theta[0] + self.theta[1] * x 
    

    def get_model(self):
        with open('model.json') as f:
            self.theta = json.load(f)

    def save_model(self):
        with open('model.json', 'w') as f:
            json.dump(self.theta, f)

    def plot_dataset(self, x, y):
        """
        Plots the dataset from x and y.
        """

        plt.figure("Prices of cars given their mileages")
        plt.plot(x, [(self.theta[0] + self.theta[1] * i) for i in x], color="r")
        plt.scatter(x, y, color="g")
        plt.xlabel("Mileages")
        plt.ylabel("Prices")
        plt.show()

    def display_cost_function(self):
        """
        Plots cost function
        """
        plt.figure("Gradient descent cost function")
        plt.plot([i for i in range(self.iteration)], self.cost, color="r")
        plt.ylabel("Cost")
        plt.xlabel("Iteration")
        plt.show()
