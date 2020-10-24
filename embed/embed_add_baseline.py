import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Baseline(nn.Module):
    def __init__(self, input_size=2, hidden_dim1=128, hidden_dim2=16, output_size=1):
        super(Baseline, self).__init__()
        self.embed = nn.Embedding(256,256)
        self.linear1 = nn.Linear(512, hidden_dim1)
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = nn.Sigmoid()
        self.linear3 = nn.Linear(hidden_dim2, 1)
        self.act3 = nn.Sigmoid()

    def forward(self, inp):
        embedded = self.embed(inp)
        flat = torch.flatten(embedded, start_dim=1)
        z_1 = self.linear1(flat)
        a_1 = self.act1(z_1)
        z_2 = self.linear2(a_1)
        a_2 = self.act2(z_2)
        z_3 = self.linear3(a_2)
        a_3 = self.act3(z_3)
        return z_3

def dataset(max_num, batch_size):

    pairs = torch.randint(max_num, (batch_size, 2))
    sums = torch.sum(pairs, dim=1) 
    sums = sums.reshape(-1, 1).float() % 256
    sums = (sums-0) / 256.

    return (torch.tensor(pairs, dtype=torch.int64), torch.tensor(sums, dtype=torch.float32))


EPOCHS=5000000
LEARNING_RATE=1e-3
BATCH_SIZE=128

model = Baseline()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

base_loss = []
base_xtra = []

if __name__ == '__main__':
    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        x_train, t_train = dataset(256, BATCH_SIZE)

        y_train = model(x_train)

        loss = criterion(y_train, t_train)
        
        base_loss.append(loss)

        if epoch % 1000 == 0:
            print("train %d: %.5f" %(epoch, loss))

        loss.backward()
        optimizer.step()
