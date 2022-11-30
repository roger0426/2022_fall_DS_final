import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class NN_sklearn_wrapper:
    def __init__(self, *args, **kwargs):
        """
        Arguments:
            in_features: input feature dimension
            n_layers: hidden layers of the NN, default=3
            dropout: dropout for NN, default=0.1
            hidden_size: size of the hidden state, default=256
        """
        self.model = BinaryClf(*args, **kwargs).to('cuda')
        self.batch_size = 64

    def np_to_torchloader(self, X, y=None, **kwargs):
        X = torch.from_numpy(X).type(torch.float)
        if y is not None:
            y = torch.from_numpy(y).type(torch.float)
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)

        dataloader = DataLoader(dataset, self.batch_size, **kwargs)
        return dataloader

    def fit(self, X, y, dev_X=None, dev_y=None, epochs=100, silent=False):
        train_loader = self.np_to_torchloader(
            X, y, shuffle=True, num_workers=4)
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        best_score = 0
        best_state_dict = None
        if not silent:
            print(f"Training on {X.shape[0]} samples")
            train_loader = tqdm(train_loader)
        for _ in range(epochs):
            for X, y in train_loader:
                X = X.to('cuda')
                y = y.to('cuda').unsqueeze(1)
                logits = self.model(X)
                loss = loss_fn(logits, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

            if any((dev_X is None, dev_y is None)):
                continue
            # Validation
            score = self.score(dev_X, dev_y, silent=True)
            if score > best_score:
                best_score = score
                best_state_dict = self.model.state_dict()
                print(f"Updated best weight with score = {best_score}")

        if best_state_dict:
            self.model.load_state_dict(best_state_dict)

    @torch.no_grad()
    def score(self, X, y, silent=False):
        """
        returns the accuracy score
        """
        datalodaer = self.np_to_torchloader(
            X, y, shuffle=False, num_workers=4)
        acc = []
        if not silent:
            print(f"Calculating score for {X.shape[0]} samples")
            datalodaer = tqdm(datalodaer)
        for X, y in datalodaer:
            X = X.to('cuda')
            y = y.to('cuda').unsqueeze(1)
            logits = self.model(X)  # shape (64, 1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).type(torch.int).squeeze()
            acc.append((preds == y).type(torch.float).mean().item())

        return sum(acc) / len(acc)

    @torch.no_grad()
    def predict_proba(self, X):
        dataloader = self.np_to_torchloader(X, shuffle=False, num_workers=4)
        all_probs = []
        print(f"Predicting probability for {X.shape[0]} samples")
        for X in tqdm(dataloader):  # X is a list with len = 1
            X = X[0].to('cuda')
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            probs = torch.cat([probs, 1 - probs], dim=1)
            all_probs.append(probs)

        return torch.cat(all_probs).cpu().numpy()

    def save_model(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class BinaryClf(nn.Module):
    def __init__(self, in_features, n_layers=3, dropout=0.1, hidden_size=256) -> None:
        super().__init__()
        layers = []
        layers.extend([
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        ])
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, 1))
        self.clf = nn.Sequential(*layers)

    def forward(self, X):
        logits = self.clf(X)
        return logits
