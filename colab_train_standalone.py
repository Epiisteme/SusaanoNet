import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# IMPORTANT: In Colab, make sure mamba3.py and model.py are in the same folder!
from core.model import HybridArchitecture, PinballLoss

# Data Preparation (Multi-Asset Tensors)
def create_multi_asset_sequences(data, seq_length, num_assets):
    xs = []
    ys = []
    
    # We want to predict the next log return for ALL 3 assets
    # In the feature dimension (14), log_return is at index 5
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        
        # Target shape: (Assets, Horizons)
        # We predict T+1 log return for all assets
        target = data[i+seq_length, :, 5] # Shape: (3,)
        ys.append(target)
        
    return np.array(xs), np.array(ys)

def load_and_scale_data(csv_path):
    print("Loading Multi-Asset CSV...")
    df = pd.read_csv(csv_path)
    
    # Drop datetime for scaling
    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])
        
    # We have 3 assets: BTC, ETH, SOL
    # 8 specific features + 6 global features = 14 features per asset
    
    btc_cols = [c for c in df.columns if c.startswith('btc_') and c != 'btc_funding_rate']
    eth_cols = [c for c in df.columns if c.startswith('eth_')]
    sol_cols = [c for c in df.columns if c.startswith('sol_')]
    
    global_cols = ['btc_funding_rate', 'sp500_close', 'dxy_close', 'gold_close', 'hash_rate', 'fng_score']
    
    # Scale everything globally first
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    # Reconstruct into (Time, Assets, Features)
    num_time = len(scaled_df)
    tensor_data = np.zeros((num_time, 3, 14))
    
    for t in range(num_time):
        row = scaled_df.iloc[t]
        
        # BTC Node
        tensor_data[t, 0, :8] = row[btc_cols].values
        tensor_data[t, 0, 8:] = row[global_cols].values
        
        # ETH Node
        tensor_data[t, 1, :8] = row[eth_cols].values
        tensor_data[t, 1, 8:] = row[global_cols].values
        
        # SOL Node
        tensor_data[t, 2, :8] = row[sol_cols].values
        tensor_data[t, 2, 8:] = row[global_cols].values
        
    return tensor_data, scaler

# Training Loop
def train_hybrid_colab():
    csv_path = 'multi_asset_15m_structural.csv'
    
    try:
        tensor_data, scaler = load_and_scale_data(csv_path)
    except Exception as e:
        print(f"Failed to load data. Make sure {csv_path} is in the Colab directory. Error: {e}")
        return
        
    print(f"Data constructed into Tensor: {tensor_data.shape} (Time, Assets, Features)")
    joblib.dump(scaler, 'hybrid_scaler.pkl')
    
    split_idx = int(len(tensor_data) * 0.8)
    train_data = tensor_data[:split_idx]
    test_data = tensor_data[split_idx:]
    
    seq_length = 60 # 15 hours of context
    
    print("Building sequences...")
    X_train, y_train = create_multi_asset_sequences(train_data, seq_length, num_assets=3)
    X_test, y_test = create_multi_asset_sequences(test_data, seq_length, num_assets=3)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Init Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 14 features in, 128 hidden dim
    model = HybridArchitecture(input_dim=14, d_model=128, d_state=64, num_assets=3, num_quantiles=3, forecast_horizons=1)
    model.to(device)
    
    criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_test_loss = float('inf')
    
    print("\nStarting Hybrid Architecture Training (VSN -> GNN -> Mamba3 -> Quantile)...")
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X) # Shape: (Batch, Assets, Horizons, Quantiles)
            
            # Target Shape: (Batch, Assets, Horizons)
            # Add horizon dimension to batch_y since we only do T+1 right now
            batch_y_h = batch_y.unsqueeze(2) 
            
            loss = criterion(outputs, batch_y_h)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                batch_y_h = batch_y.unsqueeze(2)
                loss = criterion(outputs, batch_y_h)
                test_loss += loss.item() * batch_X.size(0)
                
        test_loss /= len(test_loader.dataset)
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Pinball Loss: {train_loss:.4f} | Test Pinball Loss: {test_loss:.4f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'hybrid_mamba3_weights.pth')
            print("-> Saved new best Hybrid model!")
            
    print("Training Complete. Download hybrid_mamba3_weights.pth and hybrid_scaler.pkl!")

if __name__ == '__main__':
    train_hybrid_colab()
