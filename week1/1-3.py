import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.sgd import SGD
from torchviz import make_dot

transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)


print(len(trainset))
print(trainset[0][0].shape, trainset[0][1])
plt.imshow(trainset[0][0][0], cmap="gray")

batch_size = 128

trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

dataiter = iter(trainloader)

images, labels = next(dataiter)
print(f"🚀 : 1-3.py:24: images, labels={images.shape, labels.shape}")


class Model(nn.Module):
    def __init__(self, input_dim, n_dim):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, n_dim)
        self.layer2 = nn.Linear(n_dim, n_dim)
        self.layer3 = nn.Linear(n_dim, 1)

        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))

        return x


model = Model(28 * 28 * 1, 1024)


lr = 0.08

# mac 지원 안함
# model = model.to("cuda")

# mac에서 쓸 수 있는  gpu
(mps_device) = torch.device("mps")
model.to(mps_device)

optimizer = SGD(model.parameters(), lr=lr)

n_epochs = 24
print(f"🚀 : 1-3.py:64: bs, lr, n_epochs={batch_size, lr, n_epochs}")

for epoch in range(n_epochs):
    total_loss = 0.0

    for data in trainloader:
        model.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(mps_device), labels.to(mps_device)

        preds = model(inputs)

        loss = (preds[:, 0] - labels).pow(2).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:3d} | Loss: {total_loss}")


idx = 0

x = trainset[idx][0][None]  # (1, 1, 28, 28)
x = x.to("mps")

y = model(x)
print(y)
print(trainset[idx][1])

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("1-3-model_graph", format="png")
