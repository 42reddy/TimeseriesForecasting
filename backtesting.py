import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from scipy.stats import zscore
from ta.momentum import RSIIndicator
from sklearn.decomposition import PCA
import joblib
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import yfinance as yf
from transformer import Transformer  # Ensure this is available in your working directory


paths = ['hdfc_test.csv', 'nifty_test.csv', 'niftybank_test.csv', 'vix_test.csv']
names = ['HDFC', 'NIFTY', 'NIFTY_BANK', 'VIX']
dfs = [pd.read_csv(path) for path in paths]

for df in dfs:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)


def resample_close(df, name):
    return df['close'].resample('60T').last().rename(f'{name}_close')

resampled = {
    name: resample_close(df, name)
    for df, name in zip(dfs, names)
}

combined_df = pd.concat(resampled.values(), axis=1).dropna()
combined_df = combined_df[2000:3000]
returns_df = combined_df.pct_change().dropna()
features = pd.DataFrame(index=returns_df.index)
features['RET_HDFC'] = returns_df['HDFC_close']
features['RET_NIFTY'] = returns_df['NIFTY_close']
features['RET_DIFF_HDFC_NIFTY'] = returns_df['HDFC_close'] - returns_df['NIFTY_close']
features['VOL_HDFC'] = returns_df['HDFC_close'].rolling(12).std()
features['VIX'] = returns_df['VIX_close']
features['RSI_HDFC'] = RSIIndicator(close=combined_df['HDFC_close'].ffill(), window=14).rsi()
features['SMA_3'] = combined_df['HDFC_close'].rolling(3).mean() / combined_df['HDFC_close'] - 1
features['SMA_10'] = combined_df['HDFC_close'].rolling(10).mean() / combined_df['HDFC_close'] - 1
delta_20 = (combined_df['HDFC_close'] - combined_df['HDFC_close'].rolling(20).mean())
features['ZSCORE_HDFC_20'] = zscore(delta_20.dropna()).reindex_like(features)
features.dropna(inplace=True)

combined_df = combined_df.loc[combined_df.index.intersection(features.index)]





def generate_sequences_and_labels(f, prices_df, seq_len=50, forecast_horizon=5, lambda_decay=0.5, barrier=0.002):
    prices = prices_df['HDFC_close'].values
    X, y = [], []

    # exponential decay weights: w_k = exp(-λk)
    weights = np.exp(-lambda_decay * np.arange(1, forecast_horizon + 1))
    weights /= weights.sum()

    for i in range(len(f) - seq_len - forecast_horizon):
        x_seq = f.iloc[i:i + seq_len].values
        X.append(x_seq)

        base = i + seq_len - 1
        future_prices = prices[base + 1 : base + 1 + forecast_horizon]
        prev_prices   = prices[base     : base + forecast_horizon]

        returns = (future_prices - prev_prices) / prev_prices
        weighted_score = np.dot(weights, returns)

        if weighted_score > barrier:
            y.append(1)
        elif weighted_score < -barrier:
            y.append(-1)
        else:
            y.append(0)

    return np.array(X), np.array(y)

X, y = generate_sequences_and_labels(features, combined_df, seq_len=30, forecast_horizon=5, lambda_decay=0.25, barrier=0.001)

# Apply PCA transformation using the fitted PCA from training (pca variable from your training)
pca = joblib.load('pca_model.pkl')
X = pca.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1], 5)

def apply_pca_transform(X, fitted_pca):
    """Apply already fitted PCA transformation"""
    N, T, D = X.shape
    X_reshaped = X.reshape(-1, D)
    X_pca_flat = fitted_pca.transform(X_reshaped)
    X_pca = X_pca_flat.reshape(N, T, 5)  # n_components=5 from training
    return X_pca




def rowwise_standardize(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    return (X - means) / (stds + 1e-8)

X = rowwise_standardize(X)

X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y+1, dtype=torch.long)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_dataset = TimeSeriesDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(8,4,64, X.shape[2],128,30,3,0.5)
model.load_state_dict(torch.load('test_model.pth'))
model = model.float()
model.to(device)
model.eval()

def evaluate(model, loader, device, tag="Test"):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    accuracy = 100. * correct / total
    print(f"{tag} Accuracy on Fresh Data: {accuracy:.2f}%")
    print(f"Label distribution: {np.bincount(y_test.numpy())}")
    print(f"Prediction distribution: {np.bincount(all_preds)}")
    return accuracy, all_preds

print(f"Generated {len(X_test)} test sequences from fresh data")
print(f"Test data date range: {combined_df.index[0]} to {combined_df.index[-1]}")


accuracy, all_preds = evaluate(model, test_loader, device, tag='test')



















