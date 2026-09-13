import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from data_literal import TRAIN_DATA, VAL_DATA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# Hyperparameters

LEARNING_RATE = 0.1
w = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # [bias, teta1, teta2, teta3, teta4]
N_EPOCHS = 5

# Data -> numpy array, pisahkan fitur (X) dan target (y)
TRAIN = np.array(TRAIN_DATA)
VAL = np.array(VAL_DATA)
X_train, y_train = TRAIN[:, :4], TRAIN[:, 4]
X_val, y_val = VAL[:, :4], VAL[:, 4]


# Training + Validasi

train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(N_EPOCHS):

    # --- training ---
    squared_errors = []
    correct = 0

    for xi, yi in zip(X_train, y_train):
        z = w[0] + np.dot(w[1:], xi)
        g = 1 / (1 + np.exp(-z))          # sigmoid
        pred = 1 if g > 0.5 else 0

        error = g - yi
        squared_errors.append(error ** 2)
        correct += int(pred == yi)

        # gradient & update bobot (dipakai sampel berikutnya)
        dbias = 2 * error * (1 - g) * g
        dw = dbias * xi
        w = w - LEARNING_RATE * np.concatenate(([dbias], dw))

    train_losses.append(np.mean(squared_errors))   # MSE
    train_accs.append(correct / len(X_train))

    # --- validasi (forward pass saja, bobot = bobot akhir epoch ini) ---
    squared_errors_val = []
    correct_val = 0

    for xi, yi in zip(X_val, y_val):
        z = w[0] + np.dot(w[1:], xi)
        g = 1 / (1 + np.exp(-z))
        pred = 1 if g > 0.5 else 0

        error = g - yi
        squared_errors_val.append(error ** 2)
        correct_val += int(pred == yi)

    val_losses.append(np.mean(squared_errors_val))
    val_accs.append(correct_val / len(X_val))

    print(f"Epoch {epoch + 1}: "
          f"Train Loss={train_losses[-1]:.6f}, Train Acc={train_accs[-1]:.4f} | "
          f"Val Loss={val_losses[-1]:.6f}, Val Acc={val_accs[-1]:.4f}")

print("\nBobot akhir (bias, teta1, teta2, teta3, teta4):")
print(w)


# Simpan hasil ke JSON

results = {
    "epoch": list(range(1, N_EPOCHS + 1)),
    "train_loss": train_losses,
    "train_acc": train_accs,
    "val_loss": val_losses,
    "val_acc": val_accs,
    "final_weights": w.tolist(),
}
results_path = os.path.join(SCRIPT_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results: {results_path}")


# Grafik (smoothing tetap pakai PchipInterpolator)

epochs = results["epoch"]

# Accuracy chart
plt.figure(figsize=(7, 5))
for y_vals, label in [(train_accs, "Training Accuracy"), (val_accs, "Validation Accuracy")]:
    x_smooth = np.linspace(1, N_EPOCHS, 200)
    y_smooth = PchipInterpolator(epochs, y_vals)(x_smooth)
    line, = plt.plot(x_smooth, y_smooth, label=label)
    plt.plot(epochs, y_vals, "o", color=line.get_color())
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
for y_vals, label in [(train_losses, "Training Loss"), (val_losses, "Validation Loss")]:
    x_smooth = np.linspace(1, N_EPOCHS, 200)
    y_smooth = PchipInterpolator(epochs, y_vals)(x_smooth)
    line, = plt.plot(x_smooth, y_smooth, label=label)
    plt.plot(epochs, y_vals, "o", color=line.get_color())
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