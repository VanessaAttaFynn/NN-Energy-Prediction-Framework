import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPRegressor:
    """sklearn-style wrapper (fit/predict) around a small torch MLP.

    Standardizes features and target internally so it can be dropped into the
    same fit/predict flow as the sklearn and xgboost models without the caller
    needing separate scaling logic.
    """

    def __init__(self, hidden_sizes=(128, 64, 32), epochs=200, lr=1e-3, batch_size=256, random_state=42):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = StandardScaler().fit(y)
        X_s = self.x_scaler_.transform(X)
        y_s = self.y_scaler_.transform(y)

        self.model_ = _MLP(input_dim=X.shape[1], hidden_sizes=self.hidden_sizes)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_s), torch.from_numpy(y_s).squeeze(-1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_s = self.x_scaler_.transform(X)
        self.model_.eval()
        with torch.no_grad():
            pred_s = self.model_(torch.from_numpy(X_s)).numpy().reshape(-1, 1)
        return self.y_scaler_.inverse_transform(pred_s).ravel()


def build_model(hidden_sizes=(128, 64, 32), epochs=200, lr=1e-3, batch_size=256, random_state=42):
    return TorchMLPRegressor(
        hidden_sizes=hidden_sizes, epochs=epochs, lr=lr, batch_size=batch_size, random_state=random_state
    )
