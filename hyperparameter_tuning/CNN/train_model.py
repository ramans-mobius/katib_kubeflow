import argparse, json, time, os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# --- Parse Katib hyperparameters ---
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--dropout', type=float, required=True)
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()

# --- Dataset ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('/app/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('/app/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

# --- Training ---
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1, keepdim=True)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

# --- Write Katib metrics ---
metrics = [{
    "accuracy": f"{accuracy:.4f}",
    "loss": f"{total_loss/len(train_loader):.4f}",
    "global_step": "1",
    "timestamp": time.time(),
    "trial": "0"
}]
os.makedirs('/katib', exist_ok=True)
with open('/katib/mnist.json', "a", encoding="utf-8") as f:
    for metric in metrics:
        json.dump(metric, f)
        f.write("\n")

