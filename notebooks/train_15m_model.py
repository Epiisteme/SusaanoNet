import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys

# Ensure core module is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import RegimeAwareMambaForecaster, RealTimeHFTLoss

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        # predict the log_return of the next period (index 5)
        ys.append(data[i+seq_length, 5])
    return np.array(xs), np.array(ys)

def train_model():
    print("Loading 15m structural dataset...")
    df = pd.read_csv('../data/btc_15m_structural.csv', index_col=0)
    
    # Check features count
    features = ['open', 'high', 'low', 'close', 'volume', 'log_returns', 'regime_state',
                'volatility_1d', 'volatility_7d', 'volatility_ratio', 'skew', 'kurtosis']
    assert len(features) == 12
    
    data = df[features].values
    
    # Chronological Split (80/20)
    split_idx = int(len(data) * 0.8)
    train_data_raw = data[:split_idx]
    test_data_raw = data[split_idx:]
    
    print("Fitting Scaler on Training Data only to prevent look-ahead bias...")
    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    train_data = scaler.fit_transform(train_data_raw)
    test_data = scaler.transform(test_data_raw)
    
    # Save the scaler
    os.makedirs('../artifacts', exist_ok=True)
    joblib.dump(scaler, '../artifacts/15m_scaler.pkl')
    
    seq_length = 60
    
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    print(f"Train sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = RegimeAwareMambaForecaster(input_dim=12, d_model=64, d_state=16)
    model.to(device)
    
    criterion = RealTimeHFTLoss(direction_penalty=5.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    best_loss = float('inf')
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        correct_direction = 0
        total_direction = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                
                # Directional Accuracy calculation
                pred_sign = torch.sign(outputs)
                target_sign = torch.sign(batch_y)
                
                correct_direction += (pred_sign == target_sign).sum().item()
                total_direction += batch_y.size(0)
                
        test_loss /= len(test_loader.dataset)
        directional_acc = correct_direction / total_direction if total_direction > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Dir Acc: {directional_acc*100:.2f}%")
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), '../artifacts/15m_mamba_weights.pth')
            print(f"-> Saved new best model at Epoch {epoch+1}")
            
    print("Training Complete.")

if __name__ == '__main__':
    train_model()
