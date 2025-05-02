import torch
import torch.nn as nn

class TorchNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, output_dim=10):
        super().__init__()
        # input layer to hidden layer 1
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)  # batch normalization 

        # hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2) 

        # hidden layer 2 to output layer
        self.fc3 = nn.Linear(hidden2, output_dim)

        # activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # input layer to hidden layer 1
        x = self.relu(self.bn1(self.fc1(x))) # batch normalization

        # hidden layer 1 to hidden layer 2
        x = self.relu(self.bn2(self.fc2(x)))

        # hidden layer 2 to output layer
        return self.fc3(x)