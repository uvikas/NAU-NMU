import torch
from random import randint
import embed_add_baseline as baseline
import embed_nau as nau

nau_model = nau.model
baseline_model = baseline.model

nau_model.load_state_dict(torch.load('nau_128.opt'))
baseline_model.load_state_dict(torch.load('baseline_256.opt'))


def test_nau(a, b):
    print(a, "+", b, "=", nau_model(torch.tensor([[a, b]])).item()*256, "( expected", a+b, ")")

def test_baseline(a, b):
    print(a, "+", b, "=", baseline_model(torch.tensor([[a, b]])).item()*512, "( expected", a+b, ")")


print()
print("Testing embed_nau model")
for i in range(10):
    test_nau(randint(1, 128), randint(1, 128))

print()
print("Testing baseline model")
for i in range(10):
    test_baseline(randint(1, 128), randint(1, 128))