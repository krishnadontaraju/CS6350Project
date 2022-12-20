import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

batch_size = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DatasetImporter(Dataset):
    def __init__(self, datafile):
        raw = pd.read_csv(datafile)

        xs = raw.iloc[:, :-1].values
        ys = raw.iloc[:, -1].values

        self.xs = np.array(xs, dtype=np.float32)
        self.ys = np.array(ys, dtype=int)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    losses = []
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            losses.append(loss.item())

        if batch % 500 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct


def predict(dataloader, model):
    model.eval()
    res = np.array([])
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)
            res = np.append(res, pred.argmax(1).cpu().numpy())
    return res


DEPTH = 28
WIDTH = 50


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Sequential(nn.Linear(29, WIDTH), nn.LeakyReLU())
        self.body = nn.ModuleList([])
        for i in range(DEPTH - 2):
            self.body.append(nn.Linear(WIDTH, WIDTH))
            self.body.append(nn.LeakyReLU())
        self.out = nn.Linear(WIDTH, 2)

    def forward(self, x):
        x = self.input(x)
        for layer in self.body:
            x = layer(x)
        res = self.out(x)
        return res


train_data = DatasetImporter('trainCred.csv')
test_data = DatasetImporter('testCred.csv')

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


model = NeuralNetwork().to(device)
model.apply(init_xavier)
print(model)

weights = [1 - 0.75936, 0.75936]
class_weights = torch.FloatTensor(weights).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

best_model = model
best_acc = 0
all_losses = np.array([])

epochs = 25
for t in range(epochs):
    print(f"Epoch {t + 1}")
    losses = train(train_dataloader, model, loss_fn, optimizer)
    all_losses = np.append(all_losses, losses)
    acc = test(train_dataloader, model, loss_fn)
    if acc > best_acc:
        best_acc = acc
        best_model = model

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(all_losses, color='tab:orange', label="training loss")
ax.legend()
ax.set_xlabel("iteration")
ax.set_ylabel("Cross Entropy Loss")

plt.savefig("./images/neuraNet.png")

print(f"Best training accuracy : {(100 * best_acc):>0.1f}%")
print("Modified Neural Net Done!")