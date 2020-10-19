import torch
from random import randint
import NMU
import NAU

NMU_model = NMU.model
NAU_model = NAU.model

NMU.model.load_state_dict(torch.load('2INP_NMU_1MilEpochs.opt'))
NAU.model.load_state_dict(torch.load('2INP_NAU_1MilEpochs.opt'))

def test_NMU(a, b):
    print(a, "*", b, "=", NMU.model(torch.tensor([[a, b]])).item(), "( expected", a*b, ")")

def test_NAU(a, b):
    print(a, "+", b, "=", NAU.model(torch.tensor([[a, b]])).item(), "( expected", a+b, ")")

print("Testing NMU multiplication")
for i in range(10):
    test_NMU(float(randint(1, 100)), float(randint(1, 100)))

print()
print("Testing NAU addition")
for i in range(10):
    test_NAU(float(randint(1, 100)), float(randint(1, 100)))