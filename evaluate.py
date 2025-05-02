import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def evaluate_model(ModelClass, X_train, y_train, X_test, y_test, data_fractions):
    results = []
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    for frac in data_fractions:
        accs, times = [], []
        for _ in range(2):
            idx = np.random.choice(len(X_train), int(frac * len(X_train)), replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]
            model = ModelClass(input_dim, num_classes)
            start = time.time()
            model.train(X_sub, y_sub)
            acc = accuracy_score(y_test, model.predict(X_test))
            times.append(time.time() - start)
            accs.append(acc)
        print(f"[{ModelClass.__name__}] [{frac*100:.0f}%] Accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}, Time = {np.mean(times):.2f}s")
    return results

def evaluate_torch_model(ModelClass, X_train, y_train, X_test, y_test, data_fractions, epochs = 15):
    results = []
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    for frac in data_fractions:
        accs, times = [], []
        for _ in range(2):
            idx = np.random.choice(len(X_train), int(frac * len(X_train)), replace=False)
            X_sub = torch.tensor(X_train[idx], dtype=torch.float32)
            y_sub = torch.tensor(y_train[idx], dtype=torch.long)
            model = ModelClass(input_dim=input_dim, output_dim=num_classes)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            start = time.time()
            for _ in range(epochs):
                optimizer.zero_grad()
                output = model(X_sub)
                loss = criterion(output, y_sub)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                acc = (model(X_test_tensor).argmax(1) == y_test_tensor).float().mean().item()
            times.append(time.time() - start)
            accs.append(acc)
        print(f"[{ModelClass.__name__}] [{frac*100:.0f}%] Accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}, Time = {np.mean(times):.2f}s")
    return results