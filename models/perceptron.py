import numpy as np

class Perceptron:
    def __init__(self, input_dim, num_classes, epochs = 15):
        # weights are initialized 
        self.weights = np.zeros((num_classes, input_dim))
        self.epochs = epochs  # number of times the training set will be used to update the weights

    def train(self, X, y):
        # X is the input data, y is the corresponding labels
        # loop over the number of epochs 
        for _ in range(self.epochs):
            # loop over each example in the training set (X[i] is the ith input, y[i] is the ith label)
            for i in range(len(X)):
                # compute the scores (dot product weights and the input)
                scores = np.dot(self.weights, X[i])  
                # get the predicted class by finding the index of the maximum score (highest prediction)
                predicted = np.argmax(scores)
                
                # if the predicted class is NOT true label, update the weights
                if predicted != y[i]:
                    # update weights bc we predicted wrong
                    # add current input to correct class and subtract from predicted class
                    self.weights[y[i]] += X[i]
                    self.weights[predicted] -= X[i]

    def predict(self, X):
        # predict class labels for a given input X by calculating the dot product with weights, and returning the class with the highest score
        return np.argmax(np.dot(X, self.weights.T), axis=1)  # transpose