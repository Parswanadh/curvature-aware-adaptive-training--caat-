import argparse
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define a simple model (you can replace this with your actual model)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load data (you need to implement this based on your dataset)
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    return train_loader

# Main training function
def train(args):
    # Initialize model, optimizer, and loss function
    model = SimpleModel()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Load data
    train_loader = load_data()
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S')))
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(args.device), target.to(args.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/(batch_idx+1):.4f}')
        
        # Log training loss and learning rate
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }, f'checkpoint_{epoch + 1}.pth')
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curvature-Aware Adaptive Training Script')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (cuda or cpu).')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Interval between checkpoints.')
    
    args = parser.parse_args()
    args.device = torch.device(args.device)
    
    train(args)