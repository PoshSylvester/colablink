"""
Example: Training a simple neural network on MNIST using Colab GPU.

Run this from your local machine with:
    colablink exec python examples/train_mnist.py

The script runs on Colab GPU, but files stay local and logs appear in your terminal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train():
    """Train the model."""
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Data loading
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    print("Creating model...")
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("\nStarting training...")
    model.train()

    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/3, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(
            f"\nEpoch {epoch+1} Summary: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), "mnist_model.pt")
    print("Model saved to mnist_model.pt")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
