import torch
from caat import CAATOptimizer
from torchvision.models import resnet18

# Define your model
model = resnet18(num_classes=10)

# Define the optimizer with curvature-aware adaptive training
optimizer = CAATOptimizer(model.parameters(), lr=0.01)

# Dummy data and labels for demonstration purposes
data = torch.randn(64, 3, 224, 224)
labels = torch.randint(0, 10, (64,))

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(data)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()