import torch_geometric
print(torch_geometric.__version__)  # Should show 2.5.0+
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
df = df.dropna()

# Feature/target separation
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalization
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

# Efficient edge index construction (no self-loops)
num_nodes = X.size(0)
i, j = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
mask = (i != j)
edge_index = torch.stack([i[mask], j[mask]], dim=0)

# Create graph data object
data = Data(x=X, edge_index=edge_index, y=y)

# Train-test split (80-20)
indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_mask = torch.tensor(train_idx, dtype=torch.long)
test_mask = torch.tensor(test_idx, dtype=torch.long)

# GCN model definition
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.squeeze()

# Model initialization
model = GCN(input_dim=X.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(data)
    
y_true = data.y[test_mask].numpy()
y_pred = predictions[test_mask].numpy()

# Calculate metrics
metrics = {
    "MSE": mean_squared_error(y_true, y_pred),
    "MAE": mean_absolute_error(y_true, y_pred),
    "R²": r2_score(y_true, y_pred)
}
metrics["RMSE"] = math.sqrt(metrics["MSE"])

# Print results
print("\n--- Evaluation Metrics ---")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

