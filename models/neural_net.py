import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.01, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        # initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.biases1 = np.zeros((1, hidden_size1))

        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.biases2 = np.zeros((1, hidden_size2))

        self.weights3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.biases3 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, inputs):
        self.hidden1_output = self.sigmoid(np.dot(inputs, self.weights1) + self.biases1)
        self.hidden2_output = self.sigmoid(np.dot(self.hidden1_output, self.weights2) + self.biases2)
        self.predictions = self.sigmoid(np.dot(self.hidden2_output, self.weights3) + self.biases3)
        return self.predictions

    def backward(self, inputs, true_labels):
        num_examples = true_labels.shape[0]
        predicted_probs = self.predictions

        if true_labels.ndim == 1:
            true_labels = np.eye(predicted_probs.shape[1])[true_labels]
        output_error = predicted_probs - true_labels
        grad_weights3 = np.dot(self.hidden2_output.T, output_error) / num_examples
        grad_biases3 = np.sum(output_error, axis=0, keepdims=True) / num_examples

        hidden2_error = np.dot(output_error, self.weights3.T) * self.sigmoid_derivative(self.hidden2_output)
        grad_weights2 = np.dot(self.hidden1_output.T, hidden2_error) / num_examples
        grad_biases2 = np.sum(hidden2_error, axis=0, keepdims=True) / num_examples

        hidden1_error = np.dot(hidden2_error, self.weights2.T) * self.sigmoid_derivative(self.hidden1_output)
        grad_weights1 = np.dot(inputs.T, hidden1_error) / num_examples
        grad_biases1 = np.sum(hidden1_error, axis=0, keepdims=True) / num_examples

        # add regularization to the weight gradients
        grad_weights3 += self.lambda_reg * self.weights3
        grad_weights2 += self.lambda_reg * self.weights2
        grad_weights1 += self.lambda_reg * self.weights1

        # gradient descent
        self.weights3 -= self.learning_rate * grad_weights3
        self.biases3 -= self.learning_rate * grad_biases3

        self.weights2 -= self.learning_rate * grad_weights2
        self.biases2 -= self.learning_rate * grad_biases2

        self.weights1 -= self.learning_rate * grad_weights1
        self.biases1 -= self.learning_rate * grad_biases1

    def train(self, inputs, labels, epochs=10):
        for _ in range(epochs):
            self.forward(inputs)
            self.backward(inputs, labels)

    def predict(self, inputs):
        probabilities = self.forward(inputs)
        return np.argmax(probabilities, axis=1)