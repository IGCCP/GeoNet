import json
import matplotlib.pyplot as plt

# Load training metrics
with open('metrics/training_metrics.json', 'r') as f:
    training_metrics = json.load(f)

epochs = training_metrics['epoch']
accuracies = training_metrics['accuracy']
train_size = training_metrics['train_size']

# Plot accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies, marker='o', label=f'Training Size: {train_size}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()