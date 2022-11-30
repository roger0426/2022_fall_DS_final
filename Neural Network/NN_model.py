import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class NN_sklearn_wrapper:
    def __init__(self, *args, **kwargs):
        self.model = BinaryClf(*args, **kwargs).to('cuda')

    def fit(self, X, y):
        X = torch.from_numpy(X).type(torch.float32)
        y = torch.from_numpy(y).type(torch.int64)
        train_set = TensorDataset(X, y)
        train_loader = DataLoader(train_set, 64, shuffle=True, num_workers=4)
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(100):
            for batch in tqdm(train_loader):
                loss = self.model(batch)
                optim.zero_grad()
                loss.backward()
                optim.step()


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

    def forward(self, batch, return_logits=False):
        x, y = batch
        x, y = x.to('cuda'), y.to('cuda')
        logits = self.clf(x)
        if return_logits:
            return logits
        loss = nn.functional.cross_entropy(logits, y)
        return loss
