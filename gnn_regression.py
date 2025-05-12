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

def load_data(file_path):
    """Load dataset from Excel file, skipping metadata rows."""
    try:
        # Skip first 2 rows containing metadata/headers
        df = pd.read_excel(file_path, skiprows=2)
        return df
    except FileNotFoundError:
        print("❌ The file was not found.")
        return None

def preprocess_data(df):
    """Preprocess data and fix column names."""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Fix column name typos
    df.rename(columns={
        "Sloar Power": "Solar Power",
        "Temperature Units.1": "Temperature Units",
        "Dew Point Units.1": "Dew Point Units"
    }, inplace=True, errors='ignore')
    
    return df

def inspect_data(df):
    """Inspect data types and sample rows."""
    print("\n🔍 Data Inspection:")
    print("📊 Data Types:")
    print(df.dtypes)
    print("\n📝 First 5 Valid Data Rows:")
    print(df.head())

def prepare_features(df):
    """Prepare features and target with robust type handling."""
    features = ["Temperature Units", "Pressure Units", 
                "Relative Humidity Units", "Wind Speed Units"]
    target = "Solar Power"
    required_columns = features + [target]
    
    # Convert to numeric and handle errors
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"❌ Missing column: {col}")
            return None, None
    
    # Clean data
    df_clean = df[required_columns].dropna()
    
    if df_clean.empty:
        print("❌ No valid data remaining after cleaning")
        return None, None
    
    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    print("🔢 Final data types:")
    print(df_clean.dtypes)
    
    X = df_clean[features].values
    y = df_clean[target].values
    return X, y

def create_graph_data(X, y):
    """Create graph structure from tabular data."""
    num_nodes = X.shape[0]
    
    # Create sparse connections (every node connected to next 5 nodes)
    edge_index = []
    for i in range(num_nodes):
        for j in range(i+1, min(i+6, num_nodes)):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, y=y)

class ImprovedGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.2)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

def train_gnn_model(X, y):
    """Enhanced training loop with early stopping."""
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    
    data = create_graph_data(X, y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = ImprovedGNN(num_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    best_loss = float('inf')
    patience = 20
    counter = 0

    print("\n🚀 Training Started:")
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[idx_train], data.y[idx_train])
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
            
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f}")
            
        if counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(data.x, data.edge_index).cpu().numpy().flatten()
        y_true = data.y.cpu().numpy().flatten()
        
        print("\n📊 Final Evaluation:")
        print(f"Train samples: {len(idx_train)}")
        print(f"Test samples: {len(idx_test)}")
        
        y_pred_test = y_pred[idx_test]
        y_true_test = y_true[idx_test]
        
        mse = mean_squared_error(y_true_test, y_pred_test)
        mae = mean_absolute_error(y_true_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_test, y_pred_test)
        
        print(f"\n✅ Model Metrics:")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

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
