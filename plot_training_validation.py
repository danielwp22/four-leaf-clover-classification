import matplotlib.pyplot as plt

# Metrics extracted from your training log
train_accuracies = [75.80, 90.92, 93.68, 94.54, 96.31, 95.32, 97.94, 95.24, 98.08, 96.31]
val_accuracies = [55.20, 62.40, 56.00, 56.00, 50.40, 66.40, 50.40, 76.00, 56.00, 61.60]
num_epochs = 10

# Plot Training and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()
