import json
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from data_literal import TRAIN_DATA, VAL_DATA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Hyperparameters 

LEARNING_RATE = 0.1
INITIAL_WEIGHTS = {"bias": 0.5, "teta1": 0.5, "teta2": 0.5, "teta3": 0.5, "teta4": 0.5}
N_EPOCHS = 5



# SLP logic

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def forward(x, w):
    x1, x2, x3, x4 = x
    z = w["bias"] + w["teta1"] * x1 + w["teta2"] * x2 + w["teta3"] * x3 + w["teta4"] * x4
    gz = sigmoid(z)
    pred = 1 if gz > 0.5 else 0
    return z, gz, pred


def train_slp(data, n_epochs, lr, w):
    epoch_loss, epoch_acc, epoch_final_w = [], [], []

    for epoch in range(n_epochs):
        squared_errors, correct = [], 0
        for row in data:
            x = row[:4]
            target = row[4]

            z, gz, pred = forward(x, w)
            error = gz - target
            squared_error = error ** 2
            squared_errors.append(squared_error)
            correct += int(pred == target)

            # gradients
            dbias = 2 * (gz - target) * (1 - gz) * gz
            dteta1 = dbias * x[0]
            dteta2 = dbias * x[1]
            dteta3 = dbias * x[2]
            dteta4 = dbias * x[3]

            # update weights (used by the NEXT sample)
            w = {
                "bias":  w["bias"]  - lr * dbias,
                "teta1": w["teta1"] - lr * dteta1,
                "teta2": w["teta2"] - lr * dteta2,
                "teta3": w["teta3"] - lr * dteta3,
                "teta4": w["teta4"] - lr * dteta4,
            }

        epoch_loss.append(sum(squared_errors) / len(squared_errors))  # MSE
        epoch_acc.append(correct / len(data))
        epoch_final_w.append(dict(w))  # weights AFTER this epoch finished

    return epoch_loss, epoch_acc, epoch_final_w


def validate_slp(data, epoch_final_w):
    epoch_loss, epoch_acc = [], []

    for w in epoch_final_w:
        squared_errors, correct = [], 0
        for row in data:
            x = row[:4]
            target = row[4]
            z, gz, pred = forward(x, w)
            error = gz - target
            squared_error = error ** 2
            squared_errors.append(squared_error)
            correct += int(pred == target)

        epoch_loss.append(sum(squared_errors) / len(squared_errors))  # MSE
        epoch_acc.append(correct / len(data))

    return epoch_loss, epoch_acc


# Plotting
def smooth_plot(x, y, label):
    x = np.array(x)
    y = np.array(y)
    x_smooth = np.linspace(x.min(), x.max(), 200)
    spline = PchipInterpolator(x, y)
    y_smooth = spline(x_smooth)
    line, = plt.plot(x_smooth, y_smooth, label=label)
    plt.plot(x, y, "o", color=line.get_color())


def plot_results(results):
    epochs = results["epoch"]

    # Accuracy chart 
    plt.figure(figsize=(7, 5))
    smooth_plot(epochs, results["train_acc"], "Training Accuracy")
    smooth_plot(epochs, results["val_acc"], "Validation Accuracy")
    plt.title("Accuracy per Epoch (Training vs Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(SCRIPT_DIR, "accuracy_chart.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()

    # Loss chart 
    plt.figure(figsize=(7, 5))
    smooth_plot(epochs, results["train_loss"], "Training Loss")
    smooth_plot(epochs, results["val_loss"], "Validation Loss")
    plt.title("Loss per Epoch (Training vs Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(SCRIPT_DIR, "loss_chart.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(f"Saved chart: {acc_path}")
    print(f"Saved chart: {loss_path}")



# Main

def main():
    w0 = dict(INITIAL_WEIGHTS)

    train_loss, train_acc, epoch_final_w = train_slp(
        TRAIN_DATA, N_EPOCHS, LEARNING_RATE, w0
    )
    val_loss, val_acc = validate_slp(VAL_DATA, epoch_final_w)

    print(f"{'Epoch':<6}{'Train Loss':<14}{'Train Acc':<12}{'Val Loss':<14}{'Val Acc':<10}")
    for i in range(N_EPOCHS):
        print(
            f"{i+1:<6}{train_loss[i]:<14.6f}{train_acc[i]:<12.4f}"
            f"{val_loss[i]:<14.6f}{val_acc[i]:<10.4f}"
        )

    results = {
        "epoch": list(range(1, N_EPOCHS + 1)),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "final_weights_per_epoch": epoch_final_w,
    }
    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")

    plot_results(results)

    return results


if __name__ == "__main__":
    main()