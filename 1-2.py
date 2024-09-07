import random

import numpy as np
import torch
from torch import nn
from torch.optim.sgd import SGD

seed = 7777

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0])

print(x.shape, y.shape)


class Model(nn.Module):
    def __init__(self, d, d_prime):
        super().__init__()

        self.layer1 = nn.Linear(d, d_prime)
        self.layer2 = nn.Linear(d_prime, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (n, d)
        x = self.layer1(x)  # (n, d_prime)
        x = self.act(x)  # (n, d_prime)
        x = self.layer2(x)  # (n, 1)

        return x


model = Model(2, 10)

optimizer = SGD(model.parameters(), lr=0.1)


def train(n_epochs, model, optimizer, x, y):
    for e in range(n_epochs):
        model.zero_grad()

        y_pred = model(x)
        loss = (y_pred[:, 0] - y).pow(2).sum()

        loss.backward()
        optimizer.step()

        print(f"Epoch {e:3d} | Loss: {loss}")

    return model


n_epochs = 1
model = train(n_epochs, model, optimizer, x, y)
