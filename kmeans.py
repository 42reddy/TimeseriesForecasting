import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

paths = ['hdfc.csv', 'nifty.csv', 'niftybank.csv', 'kotak.csv', 'ril.csv', 'vix.csv']
names = ['HDFC', 'NIFTY', 'NIFTY_BANK', 'KOTAK', 'RIL', 'VIX']
dfs = [pd.read_csv(path) for path in paths]

# === Preprocess ===
for df in dfs:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

# === Resample Close Prices to 15-Min ===
def resample_close(df, name):
    return df['close'].resample('120T').last().rename(f'{name}_close')

resampled = {
    name: resample_close(df, name)
    for df, name in zip(dfs, names)
}

# === Combine all close prices ===
combined_df = pd.concat(resampled.values(), axis=1).dropna()

f = pd.DataFrame(index=combined_df.index)

# Keep only HDFC_close
f['HDFC_close'] = combined_df['HDFC_close']

# Rolling spread (NIFTY)
nifty = combined_df['NIFTY_close']
f['NIFTY_spread_50'] = nifty - nifty.rolling(7).mean()
f['NIFTY_spread_200'] = nifty - nifty.rolling(20).mean()

# RSI for HDFC
f['HDFC_RSI'] = RSIIndicator(close=combined_df['HDFC_close'].ffill(), window=14).rsi()

# Inter-stock spreads
f['SPREAD_HDFC_RIL'] = combined_df['HDFC_close'] - combined_df['RIL_close']
f['SPREAD_HDFC_KOTAK'] = combined_df['HDFC_close'] - combined_df['KOTAK_close']

f['SPREAD_HDFC_RIL'] = (f['SPREAD_HDFC_RIL'] - f['SPREAD_HDFC_RIL'].mean()) / f['SPREAD_HDFC_RIL'].std()
f['SPREAD_HDFC_KOTAK'] = (f['SPREAD_HDFC_KOTAK'] - f['SPREAD_HDFC_KOTAK'].mean()) / f['SPREAD_HDFC_KOTAK'].std()

# Returns
f['RET_HDFC'] = combined_df['HDFC_close'].pct_change()
f['RET_NIFTY'] = combined_df['NIFTY_close'].pct_change()

# Z-score of returns
f['Z_RET_HDFC'] = zscore(f['RET_HDFC'].fillna(0))
f['Z_RET_NIFTY'] = zscore(f['RET_NIFTY'].fillna(0))

f.dropna(inplace=True)
data = f
window = 5
df = pd.DataFrame(index=data.index)
df['returns'] = np.log(data['HDFCBANK.NS'] / data['HDFCBANK.NS'].shift(1)) * 100
df['mean'] = df['returns'].rolling(window).mean()
df['volatility'] = df['returns'].rolling(window).std()
f = df.dropna()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(f)

kmeans = KMeans(n_clusters=25, random_state=0)
labels_kmeans = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=0)
labels_gmm = gmm.fit_predict(X_scaled)

f['regime'] = labels_kmeans

regime_stats = f.groupby('regime').agg({
    'returns': ['mean', 'std', 'skew'],
})

means = f.groupby('regime')['returns'].mean()
vols = f.groupby('regime')['returns'].std()

plt.figure(figsize=(8, 6))
plt.scatter(vols, means, c=range(len(means)), cmap='viridis', s=100)
for i, txt in enumerate(means.index):
    plt.annotate(f'Regime {txt}', (vols[i], means[i]), fontsize=12)
plt.xlabel('Volatility (Std Dev of Returns)')
plt.ylabel('Mean Return')
plt.title('Regime Characterization: Return vs Volatility')
plt.grid(True)
plt.show()

f.groupby('regime')['returns'].std()
f.groupby('regime')['returns'].mean()

tsne = TSNE(n_components=3, perplexity=30, random_state=0)
X_2d = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_kmeans, cmap='viridis', s=10)
plt.title("t-SNE of Market Regimes via KMeans Clustering")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="Cluster Label")
plt.grid(True)
plt.show()


