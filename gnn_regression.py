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
    """Load dataset from Excel file with proper header handling."""
    try:
        # Skip first 5 rows containing metadata
        df = pd.read_excel(file_path, skiprows=5, header=None)
        
        # Manually assign column names based on your data structure
        columns = [
            'Year', 'Month', 'Day', 'Hour', 'Minute', 'Season', 
            'Cloud_Type', 'Clearsky_DHI', 'Clearsky_DNI', 'Clearsky_GHI',
            'Solar_Zenith', 'Dew_Point', 'Temperature', 'Humidity',
            'Albedo', 'Solar_Power', 'Unnamed1', 'Unnamed2', 'Unnamed3',
            'Unnamed4', 'Norm_DHI', 'Norm_DNI', 'Norm_GHI', 'Norm_Zenith',
            'Norm_Dew', 'Norm_Temp', 'Norm_Humidity', 'Norm_Albedo',
            'Norm_Solar', 'Dummy1', 'Dummy2'
        ]
        df.columns = columns[:len(df.columns)]
        return df
    except FileNotFoundError:
        print("❌ The file was not found.")
        return None

def preprocess_data(df):
    """Clean and prepare the data."""
    # Convert numeric columns
    numeric_cols = ['Temperature', 'Dew_Point', 'Humidity', 'Solar_Power']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed1', 'Unnamed2', 'Unnamed3', 'Unnamed4', 'Dummy1', 'Dummy2'])
    
    return df.dropna()

def inspect_data(df):
    """Inspect data types and sample rows."""
    print("\n🔍 Data Inspection:")
    print("📊 Data Types:")
    print(df.dtypes)
    print("\n📝 First 5 Valid Data Rows:")
    print(df.head())

def prepare_features(df):
    """Prepare features and target."""
    features = ['Temperature', 'Dew_Point', 'Humidity']
    target = 'Solar_Power'
    
    X = df[features].values
    y = df[target].values
    return X, y

def create_graph_data(X, y):
    """Create temporal graph connections."""
    num_nodes = X.shape[0]
    edge_index = []
    for i in range(num_nodes):
        # Connect each node to next 3 temporal neighbors
        for j in range(i+1, min(i+4, num_nodes)):
            edge_index.extend([[i,j], [j,i]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float).view(-1,1)
    )

class SolarGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

def train_model(X, y):
    """Training loop with early stopping."""
    data = create_graph_data(X, y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    model = SolarGNN(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    best_loss = float('inf')
    patience = 20
    counter = 0
    
    # Train-test split
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
    
    print("🚀 Training started...")
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[idx_train], data.y[idx_train])
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
            
        if counter >= patience:
            print("⏹️ Early stopping")
            break
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(data.x, data.edge_index).cpu().numpy().flatten()
        y_true = data.y.cpu().numpy().flatten()
        
        print("\n📊 Final Metrics:")
        print(f"MSE: {mean_squared_error(y_true[idx_test], y_pred[idx_test]):.4f}")
        print(f"MAE: {mean_absolute_error(y_true[idx_test], y_pred[idx_test]):.4f}")
        print(f"R²: {r2_score(y_true[idx_test], y_pred[idx_test]):.4f}")

def main():
    file_path = "Data_solar_on_27-04-2022.xlsx"
    df = load_data(file_path)
    
    if df is not None:
        df = preprocess_data(df)
        inspect_data(df)
        X, y = prepare_features(df)
        
        if X is not None and y is not None:
            train_model(X, y)

if __name__ == "__main__":
    main()
