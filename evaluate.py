import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import CurvatureAwareAdaptiveTrainingModel  # Assuming you have a model class defined in `models.py`
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Assuming normalization is applied to images normalized between -1 and 1
])

# Load the test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model and load checkpoint
model = CurvatureAwareAdaptiveTrainingModel()  # Assuming you have a constructor defined for this model
checkpoint = torch.load('path_to_your_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluation function
def evaluate():
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# Calculate evaluation metrics
accuracy, preds, labels = evaluate()
print(f'Accuracy of the model on the test images: {accuracy}%')

# Visualize results
def plot_images(images, predictions, labels, num_samples=5):
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        title = f"Pred: {predictions[i]}, True: {labels[i]}"
        plt.title(title)
        plt.axis('off')
    plt.show()

# Plot a few images to visualize the results
plot_images([test_dataset.data[i] for i in range(num_samples)], preds[:num_samples], labels[:num_samples])