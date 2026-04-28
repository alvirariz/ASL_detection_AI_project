import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



# Load data
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test  = np.load('y_test.npy')

# Convert to tensors — reshape from (N, H, W, C) to (N, C, H, W) for PyTorch
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).permute(0, 3, 1, 2)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

print("X_train tensor shape:", X_train_t.shape)
print("Ready!")

class BatchNormCNN(nn.Module):
    def __init__(self):
        super(BatchNormCNN, self).__init__()
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 29)
        )

    def forward(self, x):
        return self.fc_block(self.conv_block(x))
    

# Quick sanity check — make sure the shapes work
model = BatchNormCNN()
dummy = torch.zeros(1, 1, 64, 64)
out   = model(dummy)
print("Output shape:", out.shape)  # should be torch.Size([1, 29])
print("Model looks good!")


# Create DataLoaders — these feed data in batches during training
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=64, shuffle=False)

# Setup
model     = BatchNormCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
correct_train, total_train = 0, 0
best_acc = 0
for epoch in range(15):
    # --- Training phase ---
    model.train()
    for i, (X_batch, y_batch) in enumerate(train_loader):
     optimizer.zero_grad()
     outputs = model(X_batch)
     loss    = criterion(outputs, y_batch)
     loss.backward()
     optimizer.step()
     preds = outputs.argmax(dim=1)
     correct_train += (preds == y_batch).sum().item()
     total_train += y_batch.size(0)
     if i % 100 == 0:
         print(f"  Batch {i}/{len(train_loader)}")
    # --- Validation phase ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds    = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

    acc = correct / total * 100
    train_acc = correct_train / total_train * 100
    print(f"Epoch {epoch+1:02d}/15 | Train Acc: {train_acc:.2f}% | Val Acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
    torch.save(model.state_dict(), "best_model.pth")
    print("Saved new best model")

# Save the model
torch.save(model.state_dict(), 'model_person3.pth')
print("Saved as model_person3.pth")
