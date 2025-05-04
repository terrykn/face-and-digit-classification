import numpy as np

class NeuralNetwork:
    def __init__(self,
                 input_size,
                 hidden_size1,
                 hidden_size2,
                 output_size,
                 learning_rate=0.05,
                 lambda_reg=0.001):
        self.lr         = learning_rate
        self.lambda_reg = lambda_reg

        # He‐init for ReLU hidden layers
        self.W1 = np.random.randn(input_size,  hidden_size1) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2/hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        # Xavier‐init for softmax output
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(1/hidden_size2)
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2.dot(self.W3) + self.b3
        self.probs = self.softmax(self.Z3)
        return self.probs

    def backward(self, X, Y_onehot):
        m = X.shape[0]
        dZ3 = (self.probs - Y_onehot) / m
        dW3 = self.A2.T.dot(dZ3) + self.lambda_reg * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3.dot(self.W3.T)
        dZ2 = dA2 * self.relu_deriv(self.Z2)
        dW2 = self.A1.T.dot(dZ2) + self.lambda_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = X.T.dot(dZ1) + self.lambda_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # update
        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def train(self, X, y, epochs=15, batch_size=64):
        # turn y into one‐hot
        Y_onehot = np.eye(self.b3.shape[1])[y]
        for epoch in range(epochs):
            # shuffle
            perm = np.random.permutation(len(X))
            X_sh, Y_sh = X[perm], Y_onehot[perm]

            # mini‐batches
            for i in range(0, len(X), batch_size):
                X_batch = X_sh[i:i+batch_size]
                Y_batch = Y_sh[i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, Y_batch)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)