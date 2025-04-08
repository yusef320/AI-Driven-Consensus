import os
import glob
import csv
import math
import copy
import time

import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Helper functions
# ----------------------------

def load_pt_data(folder_path, flatten_data=True):
    """
    Given a folder containing:
        train_data.pt, train_targets.pt,
        test_data.pt, test_targets.pt
    Loads the data and returns:
      - If flatten_data is True: 
            (X_train, y_train, X_test, y_test) as flattened NumPy arrays.
      - Otherwise:
            returns the data in its original shape: [num_samples, seq_len, n_nodes]
    """
    train_data_pt = os.path.join(folder_path, "train_data.pt")
    train_targets_pt = os.path.join(folder_path, "train_targets.pt")
    test_data_pt = os.path.join(folder_path, "test_data.pt")
    test_targets_pt = os.path.join(folder_path, "test_targets.pt")

    train_data = torch.load(train_data_pt)
    train_targets = torch.load(train_targets_pt)
    test_data = torch.load(test_data_pt)
    test_targets = torch.load(test_targets_pt)

    train_data = train_data.numpy()
    train_targets = train_targets.numpy()
    test_data = test_data.numpy()
    test_targets = test_targets.numpy()

    if flatten_data:
        num_train, seq_len, n_nodes = train_data.shape
        num_test = test_data.shape[0]
        X_train = train_data.reshape(num_train, seq_len * n_nodes)
        y_train = train_targets.reshape(num_train)
        X_test = test_data.reshape(num_test, seq_len * n_nodes)
        y_test = test_targets.reshape(num_test)
        return X_train, y_train, X_test, y_test
    else:
        return train_data, train_targets, test_data, test_targets

def count_parameters(model):
    """Count the trainable parameters of a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_avg_prediction_time(model, data_loader, is_xgb=False):
    """
    Computes the average prediction time per sample using the validation data.
    For XGBoost models, data_loader is expected to be a NumPy array.
    For PyTorch models, data_loader is a DataLoader.
    """
    total_time = 0.0
    total_samples = 0
    if is_xgb:
        start = time.time()
        _ = model.predict(data_loader)
        end = time.time()
        total_time = end - start
        total_samples = data_loader.shape[0]
        return total_time / total_samples
    else:
        model.eval()
        with torch.no_grad():
            for inputs, _ in data_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _ = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                total_time += (end - start)
                total_samples += batch_size
        return total_time / total_samples if total_samples > 0 else 0.0

# ----------------------------
# Custom xLSTM Implementation
# ----------------------------

class CustomXLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_layers):
        """
        Custom implementation of an xLSTM using stacked LSTM layers with projection.
        For num_layers=1, a single LSTM layer is used.
        For num_layers>1, a projection (Linear) layer is applied between LSTM layers.
        """
        super(CustomXLSTM, self).__init__()
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        
        self.lstm_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()
        
        # First layer: from input_size to hidden_size
        self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True))
        if num_layers > 1:
            self.projection_layers.append(nn.Linear(hidden_size, proj_size))
        
        # Intermediate layers
        for layer in range(1, num_layers - 1):
            self.lstm_layers.append(nn.LSTM(input_size=proj_size, hidden_size=hidden_size, batch_first=True))
            self.projection_layers.append(nn.Linear(hidden_size, proj_size))
        
        # Last layer
        if num_layers > 1:
            self.lstm_layers.append(nn.LSTM(input_size=proj_size, hidden_size=hidden_size, batch_first=True))

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out = x
        for i in range(self.num_layers):
            out, _ = self.lstm_layers[i](out)
            if i < self.num_layers - 1:
                out = self.projection_layers[i](out)
                out = torch.relu(out)
        return out

# ----------------------------
# PyTorch Model Definitions
# ----------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class ExtendedLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, proj_size=32, num_layers=3):
        """
        Extended-LSTM model using the custom xLSTM implementation.
        """
        super(ExtendedLSTMRegressor, self).__init__()
        self.xlstm = CustomXLSTM(input_size, hidden_size, proj_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.xlstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
         super(PositionalEncoding, self).__init__()
         self.dropout = nn.Dropout(p=dropout)
         pe = torch.zeros(max_len, d_model)
         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         pe = pe.unsqueeze(0)
         self.register_buffer('pe', pe)
         
    def forward(self, x):
         x = x + self.pe[:, :x.size(1), :]
         return self.dropout(x)

class TransformerRegressor(nn.Module):
    def __init__(self, input_size, num_heads=4, hidden_dim=64, num_layers=1):
         super(TransformerRegressor, self).__init__()
         self.input_fc = nn.Linear(input_size, hidden_dim)
         encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
         self.fc_out = nn.Linear(hidden_dim, 1)
         self.positional_encoding = PositionalEncoding(hidden_dim)
         
    def forward(self, x):
         x = self.input_fc(x)
         x = self.positional_encoding(x)
         x = x.permute(1, 0, 2)
         x = self.transformer_encoder(x)
         x = x[-1, :, :]
         x = self.fc_out(x)
         return x

# ----------------------------
# Convolutional LSTM Implementation
# ----------------------------

class ConvLSTMCell1D(nn.Module):
    """
    A basic ConvLSTM cell for 1D sequences.
    Uses a 1D convolution to compute the gates.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell1D, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv1d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input, h_cur, c_cur):
        # input: (batch, input_channels, width)
        # h_cur, c_cur: (batch, hidden_channels, width)
        combined = torch.cat([input, h_cur], dim=1)  # along channel dimension
        conv_output = self.conv(combined)
        # Split into four parts for gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    """
    A simple ConvLSTM layer that processes an input sequence.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell1D(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        # x shape: (batch, seq_len, channels, width)
        batch, seq_len, channels, width = x.size()
        h = torch.zeros(batch, self.cell.hidden_channels, width, device=x.device)
        c = torch.zeros(batch, self.cell.hidden_channels, width, device=x.device)
        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :, :], h, c)
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_channels, width)
        return outputs, (h, c)

class ConvLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_channels=64, kernel_size=3):
        """
        ConvLSTM model for time series regression.
        Expects input shape (batch, seq_len, n_nodes) and reshapes it to (batch, seq_len, 1, n_nodes).
        """
        super(ConvLSTMRegressor, self).__init__()
        # Here, input_channels is 1 and width is input_size (n_nodes)
        self.convlstm = ConvLSTM(input_channels=1, hidden_channels=hidden_channels, kernel_size=kernel_size)
        # Apply adaptive average pooling over the spatial dimension (n_nodes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_nodes)
        # Insert channel dimension: (batch, seq_len, 1, n_nodes)
        x = x.unsqueeze(2)
        outputs, _ = self.convlstm(x)
        # Get the last time step output: (batch, hidden_channels, n_nodes)
        last_output = outputs[:, -1, :, :]
        # Pool over the width dimension to get (batch, hidden_channels)
        pooled = self.avg_pool(last_output).squeeze(-1)
        out = self.fc(pooled)
        return out

# ----------------------------
# Training and Evaluation Functions
# ----------------------------

def train_pytorch_model(model, train_loader, val_loader, num_epochs=150, learning_rate=1e-3, patience=30):
    """
    Trains a PyTorch model using early stopping.
    Restores the model state corresponding to the best validation loss.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def evaluate_pytorch_model(model, data_loader):
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds.append(outputs.squeeze().cpu().numpy())
            actuals.append(targets.cpu().numpy())
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    return mae, rmse, mape

# ----------------------------
# Device selection
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Main function with hyperparameter search
# ----------------------------
def main():
    data_dir = "./data_directed"
    csv_filename = "results_all_models.csv"

    # CSV header now includes "Avg Pred Time (s)" column.
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Modelo", "Case", "Graph", "Nodes", "Num Parameters",
                         "MAE", "RMSE", "MAPE", "Config", "Avg Pred Time (s)"])
        
        subfolders = sorted(glob.glob(os.path.join(data_dir, "*_seq*", "train2400_test600")))
        for folder_path in subfolders:
            parent_folder = os.path.basename(os.path.dirname(folder_path))
            parts = parent_folder.split("_")
            case_type = parts[0]
            graph_type = parts[1]
            n_nodes_str = parts[2]
            n_nodes = int(n_nodes_str[1:])
            
            if n_nodes == 2000:
                print(f"Skipping folder with n2000: {folder_path}")
                continue

            print(f"\nProcessing: {folder_path}")
            print(f"  Case: {case_type}, Graph: {graph_type}, Nodes: {n_nodes}")
            
            # ---------------------
            # XGBoost hyperparameter search (using flattened data)
            # ---------------------
            X_train_flat, y_train_flat, X_test_flat, y_test_flat = load_pt_data(folder_path, flatten_data=True)
            X_train_flat, X_val_flat, y_train_flat, y_val_flat = train_test_split(
                X_train_flat, y_train_flat, test_size=0.2, random_state=42
            )
            xgb_configs = [
                {"learning_rate": 0.1, "max_depth": 6, "n_estimators": 500},
                {"learning_rate": 0.1, "max_depth": 8, "n_estimators": 500},
                {"learning_rate": 0.1, "max_depth": 10, "n_estimators": 500},
                {"learning_rate": 0.01, "max_depth": 6, "n_estimators": 1000},
                {"learning_rate": 0.01, "max_depth": 8, "n_estimators": 1000},
                {"learning_rate": 0.01, "max_depth": 10, "n_estimators": 1000},
                {"learning_rate": 0.05, "max_depth": 7, "n_estimators": 700},
                {"learning_rate": 0.05, "max_depth": 9, "n_estimators": 700},
            ]
            best_val_mae = float("inf")
            best_xgb_model = None
            best_xgb_config = None
            for cfg in xgb_configs:
                xgb_model = xgb.XGBRegressor(
                    verbosity=1,
                    tree_method="gpu_hist",
                    predictor="gpu_predictor",
                    **cfg
                )
                xgb_model.fit(
                    X_train_flat, y_train_flat,
                    early_stopping_rounds=10,
                    eval_set=[(X_val_flat, y_val_flat)],
                    verbose=False
                )
                preds_val = xgb_model.predict(X_val_flat)
                mae_val = mean_absolute_error(y_val_flat, preds_val)
                if mae_val < best_val_mae:
                    best_val_mae = mae_val
                    best_xgb_model = xgb_model
                    best_xgb_config = cfg
            preds_xgb = best_xgb_model.predict(X_test_flat)
            mse_xgb = mean_squared_error(y_test_flat, preds_xgb)
            mae_xgb = mean_absolute_error(y_test_flat, preds_xgb)
            rmse_xgb = math.sqrt(mse_xgb)
            mape_xgb = np.mean(np.abs((y_test_flat - preds_xgb) / y_test_flat)) * 100
            avg_pred_time_xgb = compute_avg_prediction_time(best_xgb_model, X_val_flat, is_xgb=True)
            print(f"XGBoost -> MAE: {mae_xgb:.10f} | RMSE: {rmse_xgb:.10f} | MAPE: {mape_xgb:.10f}% | Avg Pred Time: {avg_pred_time_xgb:.10f} s")
            writer.writerow(["XGBoost", case_type, graph_type, n_nodes, "N/A",
                             f"{mae_xgb:.10f}", f"{rmse_xgb:.10f}", f"{mape_xgb:.10f}",
                             str(best_xgb_config), f"{avg_pred_time_xgb:.10f}"])
            
            # ---------------------
            # Prepare PyTorch datasets (original shape)
            # ---------------------
            train_data_seq, train_targets_seq, test_data_seq, test_targets_seq = load_pt_data(folder_path, flatten_data=False)
            input_size = train_data_seq.shape[2]
            train_data_seq = torch.tensor(train_data_seq, dtype=torch.float32)
            train_targets_seq = torch.tensor(train_targets_seq.squeeze(), dtype=torch.float32)
            test_data_seq = torch.tensor(test_data_seq, dtype=torch.float32)
            test_targets_seq = torch.tensor(test_targets_seq.squeeze(), dtype=torch.float32)
            full_dataset = TensorDataset(train_data_seq, train_targets_seq)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_dataset = TensorDataset(test_data_seq, test_targets_seq)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # ---------------------
            # LSTM hyperparameter search
            # ---------------------
            lstm_configs = [
                {"hidden_size": 64, "num_layers": 1},
                {"hidden_size": 128, "num_layers": 1},
                {"hidden_size": 256, "num_layers": 1},
                {"hidden_size": 64, "num_layers": 2},
                {"hidden_size": 128, "num_layers": 2},
                {"hidden_size": 256, "num_layers": 2},
                {"hidden_size": 64, "num_layers": 3},
                {"hidden_size": 128, "num_layers": 3},
            ]
            best_val_mae = float("inf")
            best_lstm_model = None
            best_lstm_config = None
            for cfg in lstm_configs:
                model = LSTMRegressor(input_size=input_size, hidden_size=cfg["hidden_size"],
                                      num_layers=cfg["num_layers"]).to(device)
                model = train_pytorch_model(model, train_loader, val_loader,
                                            num_epochs=50, learning_rate=1e-3, patience=30)
                mae_val, _, _ = evaluate_pytorch_model(model, val_loader)
                if mae_val < best_val_mae:
                    best_val_mae = mae_val
                    best_lstm_model = model
                    best_lstm_config = cfg
            mae_lstm, rmse_lstm, mape_lstm = evaluate_pytorch_model(best_lstm_model, test_loader)
            num_params_lstm = count_parameters(best_lstm_model)
            avg_pred_time_lstm = compute_avg_prediction_time(best_lstm_model, val_loader)
            print(f"LSTM -> MAE: {mae_lstm:.10f} | RMSE: {rmse_lstm:.10f} | MAPE: {mape_lstm:.10f}% | Params: {num_params_lstm} | Avg Pred Time: {avg_pred_time_lstm:.10f} s")
            writer.writerow(["LSTM", case_type, graph_type, n_nodes, num_params_lstm,
                             f"{mae_lstm:.10f}", f"{rmse_lstm:.10f}", f"{mape_lstm:.10f}",
                             str(best_lstm_config), f"{avg_pred_time_lstm:.10f}"])
            
            # ---------------------
            # Extended-LSTM hyperparameter search
            # ---------------------
            ext_lstm_configs = [
                {"hidden_size": 64, "proj_size": 32, "num_layers": 3},
                {"hidden_size": 128, "proj_size": 64, "num_layers": 3},
                {"hidden_size": 256, "proj_size": 128, "num_layers": 3},
                {"hidden_size": 64, "proj_size": 32, "num_layers": 2},
                {"hidden_size": 128, "proj_size": 64, "num_layers": 2},
                {"hidden_size": 256, "proj_size": 128, "num_layers": 2},
                {"hidden_size": 64, "proj_size": 32, "num_layers": 4},
                {"hidden_size": 128, "proj_size": 64, "num_layers": 4},
            ]
            best_val_mae = float("inf")
            best_ext_lstm_model = None
            best_ext_lstm_config = None
            for cfg in ext_lstm_configs:
                model = ExtendedLSTMRegressor(input_size=input_size, hidden_size=cfg["hidden_size"],
                                              proj_size=cfg["proj_size"], num_layers=cfg["num_layers"]).to(device)
                model = train_pytorch_model(model, train_loader, val_loader,
                                            num_epochs=50, learning_rate=1e-3, patience=30)
                mae_val, _, _ = evaluate_pytorch_model(model, val_loader)
                if mae_val < best_val_mae:
                    best_val_mae = mae_val
                    best_ext_lstm_model = model
                    best_ext_lstm_config = cfg
            mae_ext_lstm, rmse_ext_lstm, mape_ext_lstm = evaluate_pytorch_model(best_ext_lstm_model, test_loader)
            num_params_ext_lstm = count_parameters(best_ext_lstm_model)
            avg_pred_time_ext_lstm = compute_avg_prediction_time(best_ext_lstm_model, val_loader)
            print(f"Extended-LSTM -> MAE: {mae_ext_lstm:.10f} | RMSE: {rmse_ext_lstm:.10f} | MAPE: {mape_ext_lstm:.10f}% | Params: {num_params_ext_lstm} | Avg Pred Time: {avg_pred_time_ext_lstm:.10f} s")
            writer.writerow(["Extended-LSTM", case_type, graph_type, n_nodes, num_params_ext_lstm,
                             f"{mae_ext_lstm:.10f}", f"{rmse_ext_lstm:.10f}", f"{mape_ext_lstm:.10f}",
                             str(best_ext_lstm_config), f"{avg_pred_time_ext_lstm:.10f}"])
            
            # ---------------------
            # Transformer hyperparameter search
            # ---------------------
            transformer_configs = [
                {"num_heads": 4, "hidden_dim": 64, "num_layers": 1},
                {"num_heads": 4, "hidden_dim": 128, "num_layers": 1},
                {"num_heads": 8, "hidden_dim": 64, "num_layers": 1},
                {"num_heads": 8, "hidden_dim": 128, "num_layers": 1},
                {"num_heads": 4, "hidden_dim": 64, "num_layers": 2},
                {"num_heads": 4, "hidden_dim": 128, "num_layers": 2},
                {"num_heads": 8, "hidden_dim": 64, "num_layers": 2},
                {"num_heads": 8, "hidden_dim": 128, "num_layers": 2},
            ]
            best_val_mae = float("inf")
            best_transformer_model = None
            best_transformer_config = None
            for cfg in transformer_configs:
                model = TransformerRegressor(input_size=input_size, num_heads=cfg["num_heads"],
                                             hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"]).to(device)
                model = train_pytorch_model(model, train_loader, val_loader,
                                            num_epochs=50, learning_rate=1e-3, patience=30)
                mae_val, _, _ = evaluate_pytorch_model(model, val_loader)
                if mae_val < best_val_mae:
                    best_val_mae = mae_val
                    best_transformer_model = model
                    best_transformer_config = cfg
            mae_trans, rmse_trans, mape_trans = evaluate_pytorch_model(best_transformer_model, test_loader)
            num_params_trans = count_parameters(best_transformer_model)
            avg_pred_time_trans = compute_avg_prediction_time(best_transformer_model, val_loader)
            print(f"Transformer -> MAE: {mae_trans:.10f} | RMSE: {rmse_trans:.10f} | MAPE: {mape_trans:.10f}% | Params: {num_params_trans} | Avg Pred Time: {avg_pred_time_trans:.10f} s")
            writer.writerow(["Transformer", case_type, graph_type, n_nodes, num_params_trans,
                             f"{mae_trans:.10f}", f"{rmse_trans:.10f}", f"{mape_trans:.10f}",
                             str(best_transformer_config), f"{avg_pred_time_trans:.10f}"])
            
            # ---------------------
            # Convolutional LSTM hyperparameter search
            # ---------------------
            convlstm_configs = [
                {"hidden_channels": 64, "kernel_size": 3},
                {"hidden_channels": 128, "kernel_size": 3},
                {"hidden_channels": 256, "kernel_size": 3},
                {"hidden_channels": 64, "kernel_size": 5},
                {"hidden_channels": 128, "kernel_size": 5},
                {"hidden_channels": 256, "kernel_size": 5},
                {"hidden_channels": 64, "kernel_size": 7},
                {"hidden_channels": 128, "kernel_size": 7},
            ]
            best_val_mae = float("inf")
            best_conv_lstm_model = None
            best_conv_lstm_config = None
            for cfg in convlstm_configs:
                model = ConvLSTMRegressor(input_size=input_size, hidden_channels=cfg["hidden_channels"],
                                          kernel_size=cfg["kernel_size"]).to(device)
                model = train_pytorch_model(model, train_loader, val_loader,
                                            num_epochs=50, learning_rate=1e-3, patience=30)
                mae_val, _, _ = evaluate_pytorch_model(model, val_loader)
                if mae_val < best_val_mae:
                    best_val_mae = mae_val
                    best_conv_lstm_model = model
                    best_conv_lstm_config = cfg
            mae_conv, rmse_conv, mape_conv = evaluate_pytorch_model(best_conv_lstm_model, test_loader)
            num_params_conv = count_parameters(best_conv_lstm_model)
            avg_pred_time_conv = compute_avg_prediction_time(best_conv_lstm_model, val_loader)
            print(f"ConvLSTM -> MAE: {mae_conv:.10f} | RMSE: {rmse_conv:.10f} | MAPE: {mape_conv:.10f}% | Params: {num_params_conv} | Avg Pred Time: {avg_pred_time_conv:.10f} s")
            writer.writerow(["ConvLSTM", case_type, graph_type, n_nodes, num_params_conv,
                             f"{mae_conv:.10f}", f"{rmse_conv:.10f}", f"{mape_conv:.10f}",
                             str(best_conv_lstm_config), f"{avg_pred_time_conv:.10f}"])
            
    print(f"\nDone! Results saved to '{csv_filename}'")

if __name__ == "__main__":
    main()