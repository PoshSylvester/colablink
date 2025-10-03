"""Train MNIST with TensorBoard logging on Colab via ColabLink."""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class SimpleNet(nn.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
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


def train() -> None:
    """Train the model and emit TensorBoard logs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    log_dir = Path("runs") / time.strftime("mnist-%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir))

    global_step = 0
    for epoch in range(3):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

            if batch_idx % 100 == 0:
                accuracy = 100 * correct / total
                print(
                    f"Epoch {epoch + 1}/3, Batch {batch_idx}/{len(loader)}, "
                    f"Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%"
                )

        epoch_loss = running_loss / len(loader)
        epoch_accuracy = 100 * correct / total
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
        writer.add_scalar("train/epoch_accuracy", epoch_accuracy, epoch)
        print(
            f"Epoch {epoch + 1} Summary: Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )

    Path("models").mkdir(exist_ok=True)
    model_path = Path("models/mnist_tensorboard.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    writer.close()
    print(f"TensorBoard logs written to {log_dir}")
    print("Training complete!")


if __name__ == "__main__":
    train()
