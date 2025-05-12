# Install PyTorch Geometric and dependencies
import torch
import subprocess

# Detect PyTorch and CUDA versions
TORCH_VERSION = torch.__version__.split('+')[0]
CUDA_VERSION = f'cu{torch.version.cuda.replace(".", "")}' if torch.cuda.is_available() else 'cpu'

# Install required packages
subprocess.run(f"pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html", shell=True, check=True)
subprocess.run("pip install torch-geometric", shell=True, check=True)

# Now import PyG modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

# Rest of your original code remains the same
def load_data(file_path):
    """Load dataset from Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print("❌ The file was not found.")
        return None

def preprocess_data(df):
    """Preprocess data by stripping spaces from column names and renaming."""
    df.columns = df.columns.str.strip()
    df.rename(columns={"Sloar Power": "Solar Power"}, inplace=True, errors='ignore')
    return df

def inspect_data(df):
    """Inspect data types and view the first few rows."""
    print("📊 Data Types:")
    print(df.dtypes)
    print("📝 First Few Rows:")
    print(df.head())

def prepare_features(df):
    """Prepare required features and target."""
    features = ["Temperature Units", "Pressure Units", "Relative Humidity Units", "Wind Speed Units"]
    target = "Solar Power"
    required_columns = features + [target]
    
    # Attempt to convert columns to numeric
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=required_columns)
    
    # Check if any columns were excluded due to non-numeric data
    excluded_columns = [col for col in required_columns if df[col].dtype not in ['int64', 'float64']]
    if excluded_columns:
        print("❌ Excluded columns due to non-numeric data:", excluded_columns)
        return None, None
    
    print("✅ Dataframe successfully filtered with required columns!")
    X = df[features].values
    y = df[target].values
    return X, y

def create_graph_data(X, y):
    """Convert tabular data to PyTorch Geometric Data object."""
    num_nodes = X.shape[0]
    # Create fully connected graph
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index, y=y)

class GCNRegressor(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

def train_gnn_model(X, y):
    """Train a GNN regressor and evaluate metrics."""
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    
    data = create_graph_data(X, y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = GCNRegressor(num_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[idx_train], data.y[idx_train])
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(data.x, data.edge_index).cpu().numpy().flatten()
        y_true = data.y.cpu().numpy().flatten()
        
        y_pred_test = y_pred[idx_test]
        y_true_test = y_true[idx_test]
        
        mse = mean_squared_error(y_true_test, y_pred_test)
        mae = mean_absolute_error(y_true_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_test, y_pred_test)
        
        print(f"\nGNN Model MSE: {mse:.4f}")
        print(f"GNN Model MAE: {mae:.4f}")
        print(f"GNN Model RMSE: {rmse:.4f}")
        print(f"GNN Model R^2 (Coefficient of Determination): {r2:.4f}")

    return model

def main():
    file_path = "Data_solar_on_27-04-2022.xlsx"
    df = load_data(file_path)
    
    if df is not None:
        df = preprocess_data(df)
        inspect_data(df)
        X, y = prepare_features(df)
        
        if X is not None and y is not None:
            train_gnn_model(X, y)

if __name__ == "__main__":
    main()
