import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def evaluate_model(ModelClass, X_train, y_train, X_test, y_test, data_fractions, modelType):
    if modelType == "digit":
        shape = (28, 28)
    elif modelType == "face":
        shape = (70, 60)
    results = []
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    accuracies = []  # store mean accuracies for plotting
    fractions = []   # store fractions for plotting

    for frac in data_fractions:
        accs, times, preds_list = [], [], []
        
        for _ in range(2):
            idx = np.random.choice(len(X_train), int(frac * len(X_train)), replace=False)
            X_sub, y_sub = X_train[idx], y_train[idx]

            model = ModelClass(input_dim, num_classes)

            start = time.time()
            model.train(X_sub, y_sub)
            elapsed_time = time.time() - start

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            accs.append(acc)
            times.append(elapsed_time)
            preds_list.append(y_pred)

        result_entry = {
            "fraction": frac,
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "mean_time": np.mean(times),
            "predictions": preds_list  # list of arrays, one per run
        }
        results.append(result_entry)

        # store data for plotting
        accuracies.append(result_entry["mean_acc"])
        fractions.append(frac * 100)  # convert fraction to percentage

        print(f"[{modelType}][{ModelClass.__name__}][{frac*100:.0f}%] "
              f"Accuracy = {result_entry['mean_acc']:.4f} ± {result_entry['std_acc']:.4f}, "
              f"Time = {result_entry['mean_time']:.2f}s")

    # plot accuracy vs. percentage of data used
    plt.figure(figsize=(5, 4))
    plt.plot(fractions, accuracies, marker='o', label="Accuracy")
    plt.xlabel("% Training Data Used")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Training Data Use ({ModelClass.__name__})({modelType})")
    plt.grid(True)
    plt.legend()
    plt.show()

    # pick 5 random test samples
    indices = np.random.choice(len(X_test), size=5, replace=False)
    X_sample = X_test[indices]
    y_true = y_test[indices]
    y_pred = model.predict(X_sample)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        image = X_sample[i].reshape(shape)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"True: {y_true[i]}\nPred: {y_pred[i]}")
        ax.axis("off")
    plt.suptitle("Sample Predictions")
    plt.tight_layout()
    plt.show()

    
    return results


def evaluate_torch_model(ModelClass, X_train, y_train, X_test, y_test, data_fractions, modelType, epochs=15):
    if modelType == "digit":
        shape = (28, 28)
    elif modelType == "face":
        shape = (70, 60)
    results = []
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    accuracies = []  # store mean accuracies for plotting
    fractions = []   # store fractions for plotting

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
        
        result_entry = {
            "fraction": frac,
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "mean_time": np.mean(times),
        }
        results.append(result_entry)

        # store data for plotting
        accuracies.append(result_entry["mean_acc"])
        fractions.append(frac * 100)  # convert fraction to percentage

        print(f"[{modelType}][{ModelClass.__name__}][{frac*100:.0f}%] Accuracy = {result_entry['mean_acc']:.4f} ± {result_entry['std_acc']:.4f}, Time = {result_entry['mean_time']:.2f}s")

    # plot accuracy vs. percentage of data used
    plt.figure(figsize=(5, 4))
    plt.plot(fractions, accuracies, marker='o', label="Accuracy")
    plt.xlabel("% Training Data Used")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Training Data Use ({ModelClass.__name__})({modelType})")
    plt.grid(True)
    plt.legend()
    plt.show()

    # pick 5 random test samples
    indices = np.random.choice(len(X_test), size=5, replace=False)
    X_sample = torch.tensor(X_test[indices], dtype=torch.float32)
    y_true = y_test[indices]

    with torch.no_grad():
        y_pred = model(X_sample).argmax(1).numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        image = X_sample[i].numpy().reshape(shape)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"True: {y_true[i]}\nPred: {y_pred[i]}")
        ax.axis("off")
    plt.suptitle("Sample Predictions")
    plt.tight_layout()
    plt.show()

    return results


def run_model_on_random_samples(ModelClass, X_train, y_train, X_test, y_test, image_shape, title_prefix="Sample"):
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    # train the model (same setup as evaluate_model)
    model = ModelClass(input_dim, num_classes)
    model.train(X_train, y_train)

    # pick 5 random test samples
    indices = np.random.choice(len(X_test), size=5, replace=False)
    X_sample = X_test[indices]
    y_true = y_test[indices]
    y_pred = model.predict(X_sample)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        image = X_sample[i].reshape(image_shape)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{title_prefix} {i+1}\nTrue: {y_true[i]}\nPred: {y_pred[i]}")
        ax.axis("off")
    plt.suptitle("Sample Predictions")
    plt.tight_layout()
    plt.show()
