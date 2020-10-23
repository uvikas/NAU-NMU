import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Baseline(nn.Module):
    def __init__(self, input_size=1, hidden_dim=15, output_size=1):
        super(Baseline, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_dim)
        self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim, output_size)
        self.act3 = nn.Tanh()

    def forward(self, inp):
        z_1 = self.linear1(inp)
        a_1 = self.act1(z_1)
        z_2 = self.linear2(a_1)
        a_2 = self.act2(z_2)
        z_3 = self.linear3(a_2)
        a_3 = self.act3(z_3)

        return a_3

def dataset(batch_size):

    inps = torch.randint(512, (batch_size, 1)).float()
    outs = (inps % 256) / 256.

    return (torch.tensor(inps, dtype=torch.float32), torch.tensor(outs, dtype=torch.float32))


EPOCHS=5000000
LEARNING_RATE=1e-4
MOMENTUM=0.3
BATCH_SIZE=128

model = Baseline()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

base_loss = []

if __name__ == '__main__':
    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        x_train, t_train = dataset(BATCH_SIZE)

        y_train = model(x_train)

        loss = criterion(y_train, t_train)

        base_loss.append(loss)

        if epoch % 1000 == 0:
            print("train %d: %.5f" %(epoch, loss))

        loss.backward()
        optimizer.step()