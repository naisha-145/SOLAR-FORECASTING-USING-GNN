
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Load dataset
df = pd.read_excel('data/Data_solar_on_27-04-2022.xlsx')

# Drop any missing values
df = df.dropna()

# Assume last column is target (adjust if needed)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# Build edge index as fully connected (or adjust based on domain knowledge)
num_nodes = X.size(0)
row = []
col = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            row.append(i)
            col.append(j)
edge_index = torch.tensor([row, col], dtype=torch.long)

# Create PyTorch Geometric Data object
data = Data(x=X, edge_index=edge_index, y=y)

# Train-test split
train_mask, test_mask = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)
train_mask = torch.tensor(train_mask, dtype=torch.long)
test_mask = torch.tensor(test_mask, dtype=torch.long)

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze()

# Initialize model
model = GCN(input_dim=X.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(data)

# Select test values
y_true = y[test_mask].numpy()
y_pred = predictions[test_mask].numpy()

# Evaluation metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# Print metrics
print("\n--- Evaluation Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
