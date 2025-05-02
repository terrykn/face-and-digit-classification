import torch
import torch.nn as nn

class TorchNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)  # batch normalization 
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2) 
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x))) # batch normalization
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)