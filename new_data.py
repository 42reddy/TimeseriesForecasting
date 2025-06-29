import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from sklearn.decomposition import PCA
from ta.momentum import RSIIndicator
from scipy.stats import zscore
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from LSTM import lstm
from transformer import Transformer
import joblib

import warnings

warnings.filterwarnings('ignore')


paths = ['hdfc.csv', 'nifty.csv', 'niftybank.csv', 'vix.csv']
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


def apply_pca(X, n_components=5):
    N, T, D = X.shape
    X_reshaped = X.reshape(-1, D)
    pca = PCA(n_components=n_components)
    X_pca_flat = pca.fit_transform(X_reshaped)
    X_pca = X_pca_flat.reshape(N, T, n_components)
    return X_pca, pca


X, pca = apply_pca(X, 5)
joblib.dump(pca, 'pca_model.pkl')
train_size = int(len(features) * 0.95)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


def rowwise_standardize(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    return (X - means) / (stds + 1e-8)


X_train = rowwise_standardize(X_train)
X_test = rowwise_standardize(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train+1, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test+1, dtype=torch.long)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

classes = np.unique(y)

class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(8,4,64, X.shape[2],128,30,3,0.5)


criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_with_val_split(model, full_train_loader, val_split=0.2, optimizer=None, criterion=None, device='cuda', epochs=10):

    # === Split dataset ===
    full_dataset = full_train_loader.dataset
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=full_train_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=full_train_loader.batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")
        evaluate(model, val_loader, device, tag="Validation")

def evaluate(model, loader, device, tag="Eval"):
    model.eval()
    correct, total = 0, 0
    loop = tqdm(loader, desc=f"{tag}", leave=False)
    with torch.no_grad():
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            loop.set_postfix(acc=100. * correct / total if total > 0 else 0.0)
    print(f"{tag} Accuracy: {100. * correct / total:.2f}%")


train_with_val_split(model, train_loader, val_split=0.1, optimizer=optimizer, criterion=criterion, device=device, epochs=35)
























torch.save(model.state_dict(), 'test_model.pth')






device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(8,4,128, X.shape[2],128,30,3,0.5)
model.load_state_dict(torch.load('test_model.pth'))
model = model.float()
model.to(device)
model.eval()


def plot_test_metrics(model, test_loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    acc = 100. * sum([p == y for p, y in zip(all_preds, all_labels)]) / len(all_labels)
    print(f"Test Accuracy: {acc:.2f}%\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

# Example if 3-class classification
class_names = ['Down', 'Neutral', 'Up']
plot_test_metrics(model, test_loader, device, class_names)



