import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 加载实际训练数据和标签
train_data = np.load('train_data.npy')  # 替换为实际的文件路径
train_labels = np.load('train_labels.npy')  # 替换为实际的文件路径

"""
train,test 8维数据格式
[
 [1, 2, 3, 4, 5, 6, 7, 8],
 [1, 2, 3, 4, 5, 6, 7, 8],
 ...
]
train,test的label 格式
[1, 2, 3, 4, 5, ...]
"""


print("Example train_data shape:", train_data.shape)
print("Example train_labels shape:", train_labels.shape)

# Convert to Tensor type
train_tensors = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

# Create Dataset and DataLoader
train_dataset = TensorDataset(train_tensors, train_labels)


# Identify model
class StressPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StressPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # output 1 value for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Train function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for tensors, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(tensors).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    all_preds = []
    all_labels = []
    for tensors, labels in dataloader:
        outputs = model(tensors).squeeze()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f'Final MSE: {mse:.4f}, R2: {r2:.4f}')


# Dim is 8 features
input_dim = 8

# DataLoader
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = StressPredictor(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training model
num_epochs = 1000
train_model(model, train_dataloader, criterion, optimizer, num_epochs)

# Save model
model_path = 'stress_predictor_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')


