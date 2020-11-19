import torch
import math

torch.manual_seed(0)

class ReRegualizedLinearNACLayer(torch.nn.Module):
    """Implements the RegualizedLinearNAC
    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def forward(self, input, reuse=False):
        W = torch.clamp(self.W, -1.0, 1.0)
        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class ByteCombine(torch.nn.Module):
    def __init__(self, input_dim, output_dim, inner_dim=1024, **kwags):
        super().__init__()
        self.layer_1 = ReRegualizedLinearNACLayer(input_dim, inner_dim)
        self.layer_2 = ReRegualizedLinearNACLayer(inner_dim, output_dim)
        self.act = torch.nn.GELU()

        self.reset_parameters()
        self.z_1_stored = None

    def reset_parameters(self):
        self.layer_1.reset_parameters()
        self.layer_2.reset_parameters()

    def forward(self, input):
        return self.act(self.layer_2(self.act(self.layer_1(input))))