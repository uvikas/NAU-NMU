import torch
import torch.nn as nn
import os
import os.path as path
from tensorboardX import SummaryWriter as SummaryWriterRaw
import shutil
import math
import numpy as np
import collections
import itertools
import matplotlib.pyplot as plt
import time
import csv

"""
=====================================================================
NAU
=====================================================================
"""

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

torch.manual_seed(0)

class NoRandomScope:
    def __init__(self, module):
        self._module = module

    def __enter__(self):
        self._module._disable_random()

    def __exit__(self, type, value, traceback):
        self._module._enable_random()
        return False

class ExtendedTorchModule(torch.nn.Module):
    def __init__(self, default_name, *args, writer=None, name=None, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()

        self.writer = writer.namespace(default_name if name is None else name)
        self.allow_random = True

    def set_parameter(self, name, value):
        parameter = getattr(self, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            parameter.fill_(value)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.set_parameter(name, value)

    def regualizer(self, merge_in=None):
        regualizers = collections.defaultdict(int)

        if merge_in is not None:
            for key, value in merge_in.items():
                regualizers[key] += value

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                for key, value in module.regualizer().items():
                    regualizers[key] += value

        return regualizers

    def optimize(self, loss):
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.optimize(loss)

    def log_gradients(self):
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                gradient, *_ = parameter.grad.data
                #self.writer.add_summary(f'{name}/grad', gradient)
                #self.writer.add_histogram(f'{name}/grad', gradient)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_gradients()

    def no_internal_logging(self):
        return self.writer.no_logging()

    def _disable_random(self):
        self.allow_random = False
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module._disable_random()

    def _enable_random(self):
        self.allow_random = True
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module._enable_random()

    def no_random(self):
        return NoRandomScope(self)


class Regualizer:
    def __init__(self, support='nac', type='bias', shape='squared', zero=False, zero_epsilon=0):
        super()
        self.zero_epsilon = 0

        if zero:
            self.fn = self._zero
        else:
            identifier = '_'.join(['', support, type, shape])
            self.fn = getattr(self, identifier)

    def __call__(self, W):
        return self.fn(W)

    def _zero(self, W):
        return 0

    def _mnac_bias_linear(self, W):
        return torch.mean(torch.min(
            torch.abs(W - self.zero_epsilon),
            torch.abs(1 - W)
        ))

    def _mnac_bias_squared(self, W):
        return torch.mean((W - self.zero_epsilon)**2 * (1 - W)**2)

    def _mnac_oob_linear(self, W):
        return torch.mean(torch.relu(
            torch.abs(W - 0.5 - self.zero_epsilon)
            - 0.5 + self.zero_epsilon
        ))

    def _mnac_oob_squared(self, W):
        return torch.mean(torch.relu(
            torch.abs(W - 0.5 - self.zero_epsilon)
            - 0.5 + self.zero_epsilon
        )**2)

    def _nac_bias_linear(self, W):
        W_abs = torch.abs(W)
        return torch.mean(torch.min(
            W_abs,
            torch.abs(1 - W_abs)
        ))

    def _nac_bias_squared(self, W):
        return torch.mean(W**2 * (1 - torch.abs(W))**2)

    def _nac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1))

    def _nac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1)**2)

class RegualizerNMUZ:
    def __init__(self, zero=False):
        self.zero = zero
        self.stored_inputs = []

    def __call__(self, W):
        if self.zero:
            return 0

        x_mean = torch.mean(
            torch.cat(self.stored_inputs, dim=0),
            dim=0, keepdim=True
        )
        return torch.mean((1 - W) * (1 - x_mean)**2)

    def append_input(self, x):
        if self.zero:
            return
        self.stored_inputs.append(x)

    def reset(self):
        if self.zero:
            return
        self.stored_inputs = []

class RegualizerNAUZ:
    def __init__(self, zero=False):
        self.zero = zero
        self.stored_inputs = []

    def __call__(self, W):
        if self.zero:
            return 0

        x_mean = torch.mean(
            torch.cat(self.stored_inputs, dim=0),
            dim=0, keepdim=True
        )
        return torch.mean((1 - torch.abs(W)) * (0 - x_mean)**2)

    def append_input(self, x):
        if self.zero:
            return
        self.stored_inputs.append(x)

    def reset(self):
        if self.zero:
            return
        self.stored_inputs = []

def sparsity_error(W):
    W_error = torch.min(torch.abs(W), torch.abs(1 - torch.abs(W)))
    return torch.max(W_error)

def mnac(x, W, mode='prod'):
    out_size, in_size = W.size()
    x = x.view(x.size()[0], in_size, 1)
    W = W.t().view(1, in_size, out_size)

    if mode == 'prod':
        return torch.prod(x * W + 1 - W, -2)
    elif mode == 'exp-log':
        return torch.exp(torch.sum(torch.log(x * W + 1 - W), -2))
    elif mode == 'no-idendity':
        return torch.prod(x * W, -2)
    else:
        raise ValueError(f'mnac mode "{mode}" is not implemented')

class ReRegualizedLinearNACLayer(ExtendedTorchModule):
    """Implements the RegualizedLinearNAC
    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(
            support='nac', type='bias',
            shape=regualizer_shape
        )
        self._regualizer_oob = Regualizer(
            support='nac', type='oob',
            shape=regualizer_shape,
            zero=self.nac_oob == 'clip'
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        r = min(0.5, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, -r, r)

    def optimize(self, loss):
        if self.nac_oob == 'clip':
            self.W.data.clamp_(-1.0, 1.0)

    def regualizer(self):
         return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, input, reuse=False):
        W = torch.clamp(self.W, -1.0, 1.0)
        #self.writer.add_histogram('W', W)
        #self.writer.add_tensor('W', W)
        #self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        return torch.nn.functional.linear(input, W, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ReRegualizedLinearMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)
    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, regualizer_z=0,
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(
            support='mnac', type='bias',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon
        )
        self._regualizer_oob = Regualizer(
            support='mnac', type='oob',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon,
            zero=self.nac_oob == 'clip'
        )
        self._regualizer_nmu_z = RegualizerNMUZ(
            zero=regualizer_z == 0
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()

        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nmu_z(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        if self.allow_random:
            self._regualizer_nmu_z.append_input(x)

        W = torch.clamp(self.W, 0.0 + self.mnac_epsilon, 1.0) \
            if self.nac_oob == 'regualized' \
            else self.W

        if self.mnac_normalized:
            c = torch.std(x)
            x_normalized = x / c
            z_normalized = mnac(x_normalized, W, mode='prod')
            out = z_normalized * (c ** torch.sum(W, 1))
        else:
            out = mnac(x, W, mode='prod')
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


ACTIVATIONS = {
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid(),
    'ReLU6': nn.ReLU6(),
    'Softsign': nn.Softsign(),
    'SELU': nn.SELU(),
    'ELU': nn.ELU(),
    'ReLU': nn.ReLU(),
    'linear': lambda x: x,
    'GELU' : torch.nn.GELU(),
    'LeakyReLU':nn.LeakyReLU(), 
    'RandReLU':nn.RReLU(),
    'CELU': nn.CELU(),
    'Softplus':nn.Softplus(),
    'Hardshrink':nn.Hardshrink(),
    'Hardsigmoid':nn.Hardsigmoid(),
    'Hardtanh':nn.Hardtanh(),
    'Hardswish':nn.Hardswish(),
    'Tanhshrink':nn.Tanhshrink()
}

INITIALIZATIONS = {
    'Tanh': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=torch.nn.init.calculate_gain('tanh')),

    'Sigmoid': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=torch.nn.init.calculate_gain('sigmoid')),

    'ReLU6': lambda W: torch.nn.init.kaiming_uniform_(
        W, nonlinearity='relu'),

    'Softsign': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=1),

    'SELU': lambda W: torch.nn.init.uniform_(
        W, a=-math.sqrt(3/W.size(1)), b=math.sqrt(3/W.size(1))),

    # ELU: The weights have been initialized according to (He et al., 2015).
    #      source: https://arxiv.org/pdf/1511.07289.pdf
    'ELU': lambda W: torch.nn.init.kaiming_uniform_(
        W, nonlinearity='relu'),

    'ReLU': lambda W: torch.nn.init.kaiming_uniform_(
        W, nonlinearity='relu'),

    'linear': lambda W: torch.nn.init.xavier_uniform_(
        W, gain=torch.nn.init.calculate_gain('linear'))
}

class BasicLayer(ExtendedTorchModule):
    ACTIVATIONS = set(ACTIVATIONS.keys())

    def __init__(self, in_features, out_features, activation='ReLU', bias=True, **kwargs):
        super().__init__('basic', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        if activation not in ACTIVATIONS:
            raise NotImplementedError(
                f'the activation {activation} is not implemented')

        self.activation_fn = ACTIVATIONS[activation]
        self.initializer = INITIALIZATIONS[activation]

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        self.initializer(self.W)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, reuse=False):
        return self.activation_fn(
            torch.nn.functional.linear(input, self.W, self.bias)
        )

    def extra_repr(self):
        return 'in_features={}, out_features={}, activation={}'.format(
            self.in_features, self.out_features, self.activation
        )

unit_name_to_layer_class = {

    'ReRegualizedLinearMNAC': ReRegualizedLinearMNACLayer,
    'ReRegualizedLinearNAC' : ReRegualizedLinearNACLayer
}

class GeneralizedLayer(ExtendedTorchModule):
    """Abstracts all layers, both basic, NAC and NALU
    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
        unit_name: name of the unit (e.g. NAC, Sigmoid, Tanh)
    """
    UNIT_NAMES = set(unit_name_to_layer_class.keys()) | BasicLayer.ACTIVATIONS

    def __init__(self, in_features, out_features, unit_name, writer=None, name=None, **kwags):
        super().__init__('layer', name=name, writer=writer, **kwags)
        self.in_features = in_features
        self.out_features = out_features
        self.unit_name = unit_name

        if unit_name in unit_name_to_layer_class:
            Layer = unit_name_to_layer_class[unit_name]
            self.layer = Layer(in_features, out_features,
                               writer=self.writer,
                               **kwags)
        else:
            self.layer = BasicLayer(in_features, out_features,
                                    activation=unit_name,
                                    writer=self.writer,
                                    **kwags)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, input):
        return self.layer(input)

    def extra_repr(self):
        return 'in_features={}, out_features={}, unit_name={}'.format(
            self.in_features, self.out_features, self.unit_name
        )


class NAU(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, hidden_dims, act_name, unit_name='ReRegualizedLinearNAC', input_size=512, hidden_size=2, writer=None, first_layer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps

        if first_layer is not None:
            unit_name_1 = first_layer
        else:
            unit_name_1 = unit_name
        
        self.embed  = nn.Embedding(256, 256)

        #self.act1 = nn.ReLU()
        self.act_name = act_name
        self.nau_layers = []
        self.acts = []

        for i in range(len(hidden_dims)):
            if i == 0:
                self.nau_layers.append(GeneralizedLayer(input_size, hidden_dims[i], unit_name_1, writer=self.writer, name='layer_%d' %i, eps=eps, **kwags))
            else:
                self.nau_layers.append(GeneralizedLayer(hidden_dims[i-1], hidden_dims[i], unit_name_1, writer=self.writer, name='layer_%d' %i, eps=eps, **kwags))
            if self.act_name != 'linear':
                self.acts.append(ACTIVATIONS[act_name])

        self.nau_layers.append(GeneralizedLayer(hidden_dims[len(hidden_dims)-1], 1, unit_name_1, writer=self.writer, name='layer_%d' %(len(hidden_dims)), eps=eps, **kwags))
        if self.act_name != 'linear':
            self.acts.append(ACTIVATIONS[act_name])



        """
        self.layer_1 = GeneralizedLayer(input_size, 128,
                                        unit_name_1,
                                        writer=self.writer,
                                        name='layer_1',
                                        eps=eps, **kwags)

        if nac_mul == 'mnac':
            unit_name_2 = unit_name[0:-3] + 'MNAC'
        else:
            unit_name_2 = unit_name
            
        self.act2 = nn.ReLU()

        self.layer_2 = GeneralizedLayer(128, 1,
                                        'linear' if unit_name_2 in BasicLayer.ACTIVATIONS else unit_name_2,
                                        writer=self.writer,
                                        name='layer_2',
                                        eps=eps, **kwags)
        
        self.act3 = nn.ReLU()
    
        self.layer_3 = GeneralizedLayer(16, 1,
                                        'linear' if unit_name_2 in BasicLayer.ACTIVATIONS else unit_name_2,
                                        writer=self.writer,
                                        name='layer_3',
                                        eps=eps, **kwags)

        self.act4 = nn.ReLU()

        self.layer_4 = nn.Linear(4, 1)
        """

        
        self.reset_parameters()
        self.z_1_stored = None

    def reset_parameters(self):
        for i in range(len(self.nau_layers)):
            self.nau_layers[i].reset_parameters()


        #self.layer_1.reset_parameters()
        #self.layer_2.reset_parameters()

    def regualizer(self):
        if self.nac_mul == 'max-safe':
            return super().regualizer({
                'z': torch.mean(torch.relu(1 - self.z_1_stored))
            })
        else:
            return super().regualizer()
    
    def get_model_size(self):
        # Input
        input_size = (1,1,2)
        input_bits = np.prod(input_size)*64

        # Parameters
        param_bits = 0
        param_bits += np.prod(np.array(list(self.embed.parameters())[0].size()))*32
        for lay in self.nau_layers:
            param_bits += np.prod(np.array(list(lay.parameters())[0].size()))*32

        # Forward Backward
        forw_back_bits = 0
        ex_inp = torch.tensor([[1, 1]])
        out = self.embed(ex_inp)
        forw_back_bits += np.prod(np.array(out.size()))*32*2
        out = torch.flatten(out, start_dim=1)
        forw_back_bits += np.prod(np.array(out.size()))*32
        for i in range(len(self.nau_layers)):
            out = self.nau_layers[i](out)
            forw_back_bits += np.prod(np.array(out.size()))*32*2
            if self.act_name != 'linear':
                out = self.acts[i](out)
                forw_back_bits += np.prod(np.array(out.size()))*32

        return (input_bits, param_bits, forw_back_bits)


    def forward(self, input):

        embedded = self.embed(input)
        #print("Embed shape:", embedded.shape)
        out = torch.flatten(embedded, start_dim=1)
        #print("Flatten shape:", out.shape)

        for i in range(len(self.nau_layers)):
            #if self.act_name != 'linear':
                #out = self.acts[i](out)
            out = self.nau_layers[i](out)
            if i == 0:
                out = self.acts[i](out)

            #print("Layer", i+1, ":", out.shape)
        #out = self.acts[0](out)

        return out

        """

        #self.writer.add_summary('x', input)

        embedded = self.embed(input)
        self.writer.add_summary('embed', embedded)

        self.writer.add_tensor('embedded', self.embed.weight)

        #embedde = self.act(embedded)

        flat = torch.flatten(embedded, start_dim=1)

        z_1 = self.layer_1(flat)
        a_1 = self.act1(z_1)
        
        self.z_1_stored = z_1
        self.writer.add_summary('z_1', z_1)
        
        #z_2 = self.layer_2(a_1)
        #a_2 = self.act2(z_2)
        
        #z_3 = self.layer_3(a_2)
        #a_3 = self.act3(z_3)
        if self.nac_mul == 'none' or self.nac_mul == 'mnac':
            z_2 = self.layer_2(a_1)
            a_2 = self.act2(z_2)
            #z_3 = self.layer_3(z_2)
            #a_3 = self.act3(z_3)
            #z_4 = self.layer_4(z_3)
        elif self.nac_mul == 'normal':
            z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1) + self.eps)))
        elif self.nac_mul == 'safe':
            z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1 - 1) + 1)))
        elif self.nac_mul == 'max-safe':
            z_2 = torch.exp(self.layer_2(torch.log(torch.relu(z_1 - 1) + 1)))
        else:
            raise ValueError(f'Unsupported nac_mul option ({self.nac_mul})')

        self.writer.add_summary('z_2', z_2)

        return a_2

        """

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )




THIS_DIR = path.dirname(path.realpath(__file__))

if 'TENSORBOARD_DIR' in os.environ:
    TENSORBOARD_DIR = os.environ['TENSORBOARD_DIR']
else:
    TENSORBOARD_DIR = path.join(THIS_DIR, '../../../tensorboard')

class SummaryWriterNamespaceNoLoggingScope:
    def __init__(self, writer):
        self._writer = writer

    def __enter__(self):
        self._writer._logging_enabled = False

    def __exit__(self, type, value, traceback):
        self._writer._logging_enabled = True
        return False

class DummySummaryWriter():
    def __init__(self, **kwargs):
        self._logging_enabled = False
        pass

    def add_scalar(self, name, value, verbose_only=True):
        pass

    def add_summary(self, name, tensor, verbose_only=True):
        pass

    def add_histogram(self, name, tensor, verbose_only=True):
        pass

    def add_tensor(self, name, tensor, verbose_only=True):
        pass

    def print(self, name, tensor, verbose_only=True):
        pass

    def namespace(self, name):
        return self

    def every(self, epoch_interval):
        return self

    def verbose(self, verbose):
        return self

    def no_logging(self):
        return SummaryWriterNamespaceNoLoggingScope(self)

class SummaryWriterNamespace:
    def __init__(self, namespace='', epoch_interval=1, verbose=True, root=None, parent=None):
        self._namespace = namespace
        self._epoch_interval = epoch_interval
        self._verbose = verbose
        self._parent = parent
        self._logging_enabled = True
        self._force_logging = False

        if root is None:
            self._root = self
        else:
            self._root = root

    def get_iteration(self):
        return self._root.get_iteration()

    def is_log_iteration(self):
        return (self._root.get_iteration() % self._epoch_interval == 0) or self._root._force_logging

    def is_logging_enabled(self):
        writer = self
        while writer is not None:
            if writer._logging_enabled:
                writer = writer._parent
            else:
                return False
        return True

    def is_verbose(self, verbose_only):
        return (verbose_only is False or self._verbose)

    def add_scalar(self, name, value, verbose_only=True):
        if self.is_log_iteration() and self.is_logging_enabled() and self.is_verbose(verbose_only):
            self._root.writer.add_scalar(f'{self._namespace}/{name}', value, self.get_iteration())

    def add_summary(self, name, tensor, verbose_only=True):
        if self.is_log_iteration() and self.is_logging_enabled() and self.is_verbose(verbose_only):
            self._root.writer.add_scalar(f'{self._namespace}/{name}/mean', torch.mean(tensor), self.get_iteration())
            self._root.writer.add_scalar(f'{self._namespace}/{name}/var', torch.var(tensor), self.get_iteration())

    def add_tensor(self, name, matrix, verbose_only=True):
        if self.is_log_iteration() and self.is_logging_enabled() and self.is_verbose(verbose_only):
            data = matrix.detach().cpu().numpy()
            data_str = np.array2string(data, max_line_width=60, threshold=np.inf)
            self._root.writer.add_text(f'{self._namespace}/{name}', f'<pre>{data_str}</pre>', self.get_iteration())

    def add_histogram(self, name, tensor, verbose_only=True):
        if torch.isnan(tensor).any():
            print(f'nan detected in {self._namespace}/{name}')
            tensor = torch.where(torch.isnan(tensor), torch.tensor(0, dtype=tensor.dtype), tensor)
            raise ValueError('nan detected')

        if self.is_log_iteration() and self.is_logging_enabled() and self.is_verbose(verbose_only):
            self._root.writer.add_histogram(f'{self._namespace}/{name}', tensor, self.get_iteration())

    def print(self, name, tensor, verbose_only=True):
        if self.is_log_iteration() and self.is_logging_enabled() and self.is_verbose(verbose_only):
            print(f'{self._namespace}/{name}:')
            print(tensor)

    def namespace(self, name):
        return SummaryWriterNamespace(
            namespace=f'{self._namespace}/{name}',
            epoch_interval=self._epoch_interval,
            verbose=self._verbose,
            root=self._root,
            parent=self,
        )

    def every(self, epoch_interval):
        return SummaryWriterNamespace(
            namespace=self._namespace,
            epoch_interval=epoch_interval,
            verbose=self._verbose,
            root=self._root,
            parent=self,
        )

    def verbose(self, verbose):
        return SummaryWriterNamespace(
            namespace=self._namespace,
            epoch_interval=self._epoch_interval,
            verbose=verbose,
            root=self._root,
            parent=self,
        )

    def no_logging(self):
        return SummaryWriterNamespaceNoLoggingScope(self)

    def force_logging(self, flag):
        return SummaryWriterNamespaceForceLoggingScope(self, flag)

class SummaryWriter(SummaryWriterNamespace):
    def __init__(self, name, remove_existing_data=False, **kwargs):
        super().__init__()
        self.name = name
        self._iteration = 0

        log_dir = path.join(TENSORBOARD_DIR, name)
        if path.exists(log_dir) and remove_existing_data:
            shutil.rmtree(log_dir)

        self.writer = SummaryWriterRaw(log_dir=log_dir, **kwargs)

    def set_iteration(self, iteration):
        self._iteration = iteration

    def get_iteration(self):
        return self._iteration

    def close(self):
        self.writer.close()

    def __del__(self):
        self.close()

class ARITHMETIC_FUNCTIONS_STRINGIY:
    @staticmethod
    def add(*subsets):
        return ' + '.join(map(str, subsets))

    @staticmethod
    def sub(a, b, *extra):
        return f'{a} - {b}'

    @staticmethod
    def mul(*subsets):
        return ' * '.join(map(str, subsets))

    def div(a, b):
        return f'{a} / {b}'

    def squared(a, *extra):
        return f'{a}**2'

    def root(a, *extra):
        return f'sqrt({a})'

class ARITHMETIC_FUNCTIONS:
    @staticmethod
    def add(*subsets):
        return np.sum(subsets, axis=0)

    @staticmethod
    def sub(a, b, *extra):
        return a - b

    @staticmethod
    def mul(*subsets):
        return np.prod(subsets, axis=0)

    def div(a, b, *extra):
        return a / b

    def squared(a, *extra):
        return a * a

    def root(a, *extra):
        return np.sqrt(a)

class FastDataLoader:
    def __init__(self, dataset, batch_size, use_cuda):
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def __iter__(self):
        for i in range(len(self)):
            values = self.dataset[i * self.batch_size: min(len(self.dataset), (1 + i)*self.batch_size)]
            if self.use_cuda:
                yield tuple(value.cuda() for value in values)
            else:
                yield values

    def __len__(self):
        return len(self.dataset) // self.batch_size

class SimpleFunctionDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, shape, sample_range, rng):
        super().__init__()

        if not isinstance(sample_range[0], list):
            sample_range = [sample_range]
        else:
            if (sample_range[0][0] - sample_range[0][1]) != (sample_range[1][0] - sample_range[1][1]):
                raise ValueError(f'unsymetric range for {sample_range}')

        self._shape = shape
        self._sample_range = sample_range
        self._rng = rng

        self._operation = parent._operation
        self._input_size = parent._input_size
        self._max_size = parent._max_size
        self._use_cuda = parent._use_cuda

        self._subset_ranges = parent.subset_ranges

    def _multi_uniform_sample(self, batch_size):
        if len(self._sample_range) == 1:
            return self._rng.uniform(
                low=self._sample_range[0][0],
                high=self._sample_range[0][1],
                size=(batch_size, ) + self._shape)
        elif len(self._sample_range) == 2:
            part_0 = self._rng.uniform(
                low=self._sample_range[0][0],
                high=self._sample_range[0][1],
                size=(batch_size, ) + self._shape)

            part_1 = self._rng.uniform(
                low=self._sample_range[1][0],
                high=self._sample_range[1][1],
                size=(batch_size, ) + self._shape)

            choose = self._rng.randint(
                2,
                size=(batch_size, ) + self._shape)

            return np.where(choose, part_0, part_1)
        else:
            raise NotImplemented()

    def __getitem__(self, select):
        # Assume select represent a batch_size by using self[0:batch_size]
        batch_size = select.stop - select.start if isinstance(select, slice) else 1

        input_vector = self._multi_uniform_sample(batch_size)

        #print("START")
        #print(embed)
        pairs = torch.randint(256, (batch_size, 2))
        #print(pairs[0])
        sums = torch.sum(pairs, dim=1) % 256
        sums = sums.reshape(-1, 1).float()
        sums = (sums-0) / 256.
        #print(pairs[0])

        #print(inp_vec[0])
        #print(sums[0])

        #output_vec = torch.FloatTensor(batch_size, 256)
        #output_vec.zero_()
        #output_vec.scatter_(1, sums, 1)

        #print("START")
        #print(inp_vec.shape)
        #print(sums.shape)


        #a, b = random.sample(range(1, 127), 2)
        #s = a+b

        #a = embed(torch.LongTensor([a]))
        #b = embed(torch.LongTensor([b]))

        #inp_vec = torch.cat(a, b, 1)
        #out_vec = torch.tensor(s, dtype=torch.float32)


        # Compute a and b values
        
        sum_axies = tuple(range(1, 1 + len(self._shape)))
        subsets = [
            np.sum(input_vector[..., start:end], axis=sum_axies)
            for start, end in self._subset_ranges
        ]

        # Compute result of arithmetic operation
        output_scalar = self._operation(*subsets)[:, np.newaxis]

        # If select is an index, just return the content of one row
        if not isinstance(select, slice):
            input_vector = input_vector[0]
            output_scalar = output_scalar[0]
        """
        print("BEGIN")
        print(batch_size)
        print(self._subset_ranges)
        print(input_vector.shape)
        print(sum_axies)
        print("SUBSETS SHAPE:", len(subsets))
        print("SUBSETS SHAPE:", subsets[0].shape)
        print(output_scalar)"""

        #print(inp_vec.shape)
        #print(embed(sums).shape)

        return (
            pairs, sums
        )

    def __len__(self):
        return self._max_size

    def dataloader(self, batch_size=128):
        return FastDataLoader(self, batch_size, self._use_cuda)

class SimpleFunctionDataset:
    def __init__(self, operation, input_size,
                 subset_ratio=0.25,
                 overlap_ratio=0.5,
                 num_subsets=2,
                 simple=True,
                 seed=None,
                 use_cuda=False,
                 max_size=2**32-1):
        super().__init__()
        self._operation_name = operation
        self._operation = getattr(ARITHMETIC_FUNCTIONS, operation)
        self._max_size = max_size
        self._use_cuda = use_cuda
        self._rng = np.random.RandomState(seed)

        if simple:
            self._input_size = 2

            self.subset_ranges = [(0, 4), (0, 2)]
        else:
            self._input_size = input_size
            subset_size = math.floor(subset_ratio * input_size)
            overlap_size = math.floor(overlap_ratio * subset_size)

            self.subset_ranges = []
            for subset_i in range(num_subsets):
                start = 0 if subset_i == 0 else self.subset_ranges[-1][1] - overlap_size
                end = start + subset_size
                self.subset_ranges.append((start, end))

            
            total_used_size = self.subset_ranges[-1][1]
            if total_used_size > input_size:
                raise ValueError('too many subsets given the subset and overlap ratios')

            offset = self._rng.randint(0, input_size - total_used_size + 1)
            self.subset_ranges = [
                (start + offset, end + offset)
                for start, end in self.subset_ranges
            ]

            print(self.subset_ranges)

    def print_operation(self):
        subset_str = [
            f'sum(v[{start}:{end}])' for start, end in self.subset_ranges
        ]
        return getattr(ARITHMETIC_FUNCTIONS_STRINGIY, self._operation_name)(*subset_str)

    def get_input_size(self):
        return self._input_size

    def fork(self, shape, sample_range, seed=None):
        assert shape[-1] == self._input_size

        rng = np.random.RandomState(self._rng.randint(0, 2**32 - 1) if seed is None else seed)
        return SimpleFunctionDatasetFork(self, shape, sample_range, rng)

class SimpleFunctionStaticDataset(SimpleFunctionDataset):
    def __init__(self, operation,
                 input_size=100,
                 **kwargs):
        super().__init__(operation, input_size,
                         **kwargs)

    def fork(self, sample_range=[1, 2], *args, **kwargs):
        return super().fork((self._input_size, ), sample_range, *args, **kwargs)


LAYER_TYPE='ReRegualizedLinearNAC'
OPERATION='add'
NUM_SUBSETS=2
REGUALIZER=10
REGUALIZER_Z=0
REGUALIZER_OOB=1
FIRST_LAYER=None
MAX_ITERATIONS=5000000
BATCH_SIZE=128
SEED=0
INTERPOLATION_RANGE=[1,256]
EXTRAPOLATION_RANGE=[1,256]
INPUT_SIZE=512
SUBSET_RATIO=1
OVERLAP_RATIO=1
SIMPLE=True
HIDDEN_SIZE=100
NAC_MUL='none'
OOB_MODE='clip'
REGUALIZER_SCALING='linear'
REGUALIZER_SCALING_START=4000
REGUALIZER_SCALING_END=2000000
REGUALIZER_SHAPE='linear'
MNAC_EPSILON=0
NALU_BIAS=False
NALU_TWO_MNAC=False
NALU_TWO_GATE=False
NALU_MUL=False
NALU_GATE='normal'
OPTIMIZER='adam'
LEARNING_RATE=1e-3
MOMENTUM=0.5  #0.2
NO_CUDA=False
NAME_PREFIX='NAU'
REMOVE_EXISTING_DATA=True
VERBOSE=False
MINI_BATCH_SIZE=16

summary_writer = SummaryWriter(
        'NAU',
        remove_existing_data=REMOVE_EXISTING_DATA
)

dataset = SimpleFunctionStaticDataset(
    operation=OPERATION,
    input_size=INPUT_SIZE,
    subset_ratio=SUBSET_RATIO,
    overlap_ratio=OVERLAP_RATIO,
    num_subsets=NUM_SUBSETS,
    simple=SIMPLE,
    seed=0,
)

# ReLU GELU linear
"""
model = NAU(
    (100,20),
    'ReLU',
    unit_name=LAYER_TYPE,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    nac_oob=OOB_MODE,
    regualizer_shape=REGUALIZER_SHAPE,
    regualizer_z=REGUALIZER_Z,
    mnac_epsilon=MNAC_EPSILON,
    writer=summary_writer.every(1000).verbose(VERBOSE),
    nac_mul=NAC_MUL,
)
"""

def train_nau(model, epochs):
    model.reset_parameters()

    dataset_train = iter(dataset.fork(sample_range=INTERPOLATION_RANGE).dataloader(batch_size=BATCH_SIZE))
    dataset_valid_interpolation_data = next(iter(dataset.fork(sample_range=INTERPOLATION_RANGE).dataloader(batch_size=10000)))
    dataset_test_extrapolation_data = next(iter(dataset.fork(sample_range=EXTRAPOLATION_RANGE).dataloader(batch_size=10000)))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


    def test_model(data):
        with torch.no_grad(), model.no_internal_logging(), model.no_random():
            x, t = data
            return criterion(model(x), t)


    # Train model

    losses = []
    for epoch_i, (x_train, t_train) in zip(range(epochs + 1), dataset_train):
        mini_batch_loss = []
        for mini in range(MINI_BATCH_SIZE):
            summary_writer.set_iteration(epoch_i)

            # Prepear model
            model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
            optimizer.zero_grad()

            # Log validation
            if epoch_i % 1000 == 0:
                interpolation_error = test_model(dataset_valid_interpolation_data)
                extrapolation_error = test_model(dataset_test_extrapolation_data)

                summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
                summary_writer.add_scalar('metric/test/extrapolation', extrapolation_error)

            # forward
            y_train = model(x_train)
            regualizers = model.regualizer()

            if (REGUALIZER_SCALING == 'linear'):
                r_w_scale = max(0, min(1, (
                    (epoch_i - REGUALIZER_SCALING_START) /
                    (REGUALIZER_SCALING_END - REGUALIZER_SCALING_START)
                )))
            elif (REGUALIZER_SCALING == 'exp'):
                r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

            loss_train_criterion = criterion(y_train, t_train)
            mini_batch_loss.append(loss_train_criterion)
            loss_train_regualizer = REGUALIZER * r_w_scale * regualizers['W'] + regualizers['g'] + REGUALIZER_Z * regualizers['z'] + REGUALIZER_OOB * regualizers['W-OOB']
            loss_train = loss_train_criterion

            # Log loss
            if VERBOSE or epoch_i % 1000 == 0:
                summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
                summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
                summary_writer.add_scalar('loss/train/total', loss_train)
            #if epoch_i % 1000 == 0:
            #    print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

            # Optimize model
            if loss_train.requires_grad:
                loss_train.backward()
                optimizer.step()
            model.optimize(loss_train_criterion)

            # Log gradients if in verbose mode
            if VERBOSE and epoch_i % 1000 == 0:
                model.log_gradients()
            
            #print(epoch_i, mini, loss_train.item())
        
        losses.append(sum(mini_batch_loss)/len(mini_batch_loss))
        if(epoch_i % 500 == 0):
            print('train %d: %.5f' % (epoch_i, sum(mini_batch_loss)/len(mini_batch_loss)))

    return smooth(losses, 5)


def train_nau_reg(model, epochs):
    model.reset_parameters()

    dataset_train = iter(dataset.fork(sample_range=INTERPOLATION_RANGE).dataloader(batch_size=BATCH_SIZE))
    dataset_valid_interpolation_data = next(iter(dataset.fork(sample_range=INTERPOLATION_RANGE).dataloader(batch_size=10000)))
    dataset_test_extrapolation_data = next(iter(dataset.fork(sample_range=EXTRAPOLATION_RANGE).dataloader(batch_size=10000)))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


    def test_model(data):
        with torch.no_grad(), model.no_internal_logging(), model.no_random():
            x, t = data
            return criterion(model(x), t)


    # Train model

    losses = []
    for epoch_i, (x_train, t_train) in zip(range(epochs + 1), dataset_train):
        mini_batch_loss = []
        for mini in range(MINI_BATCH_SIZE):
            summary_writer.set_iteration(epoch_i)

            # Prepear model
            model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
            optimizer.zero_grad()

            # Log validation
            if epoch_i % 1000 == 0:
                interpolation_error = test_model(dataset_valid_interpolation_data)
                extrapolation_error = test_model(dataset_test_extrapolation_data)

                summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
                summary_writer.add_scalar('metric/test/extrapolation', extrapolation_error)

            # forward
            y_train = model(x_train)
            regualizers = model.regualizer()

            if (REGUALIZER_SCALING == 'linear'):
                r_w_scale = max(0, min(1, (
                    (epoch_i - REGUALIZER_SCALING_START) /
                    (REGUALIZER_SCALING_END - REGUALIZER_SCALING_START)
                )))
            elif (REGUALIZER_SCALING == 'exp'):
                r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

            loss_train_criterion = criterion(y_train, t_train)
            mini_batch_loss.append(loss_train_criterion)
            loss_train_regualizer = REGUALIZER * r_w_scale * regualizers['W'] + regualizers['g'] + REGUALIZER_Z * regualizers['z'] + REGUALIZER_OOB * regualizers['W-OOB']
            loss_train = loss_train_criterion + loss_train_regualizer

            # Log loss
            if VERBOSE or epoch_i % 1000 == 0:
                summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
                summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
                summary_writer.add_scalar('loss/train/total', loss_train)
            #if epoch_i % 1000 == 0:
            #    print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

            # Optimize model
            if loss_train.requires_grad:
                loss_train.backward()
                optimizer.step()
            model.optimize(loss_train_criterion)

            # Log gradients if in verbose mode
            if VERBOSE and epoch_i % 1000 == 0:
                model.log_gradients()
            
            #print(epoch_i, mini, loss_train.item())
        
        losses.append(sum(mini_batch_loss)/len(mini_batch_loss))
        if(epoch_i % 500 == 0):
            print('train %d: %.5f' % (epoch_i, sum(mini_batch_loss)/len(mini_batch_loss)))

    return smooth(losses, 5)

"""
if __name__ == '__main__':

    model.reset_parameters()
model = NAU(
    (100,20),
    'ReLU',
    unit_name=LAYER_TYPE,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    nac_oob=OOB_MODE,
    regualizer_shape=REGUALIZER_SHAPE,
    regualizer_z=REGUALIZER_Z,
    mnac_epsilon=MNAC_EPSILON,
    writer=summary_writer.every(1000).verbose(VERBOSE),
    nac_mul=NAC_MUL,
)
    dataset_train = iter(dataset.fork(sample_range=INTERPOLATION_RANGE).dataloader(batch_size=BATCH_SIZE))
    dataset_valid_interpolation_data = next(iter(dataset.fork(sample_range=INTERPOLATION_RANGE).dataloader(batch_size=10000)))
    dataset_test_extrapolation_data = next(iter(dataset.fork(sample_range=EXTRAPOLATION_RANGE).dataloader(batch_size=10000)))

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


    def test_model(data):
        with torch.no_grad(), model.no_internal_logging(), model.no_random():
            x, t = data
            return criterion(model(x), t)


    # Train model
    print('')
    for epoch_i, (x_train, t_train) in zip(range(MAX_ITERATIONS + 1), dataset_train):
        summary_writer.set_iteration(epoch_i)

        # Prepear model
        model.set_parameter('tau', max(0.5, math.exp(-1e-5 * epoch_i)))
        optimizer.zero_grad()

        # Log validation
        if epoch_i % 1000 == 0:
            interpolation_error = test_model(dataset_valid_interpolation_data)
            extrapolation_error = test_model(dataset_test_extrapolation_data)

            summary_writer.add_scalar('metric/valid/interpolation', interpolation_error)
            summary_writer.add_scalar('metric/test/extrapolation', extrapolation_error)

        # forward
        y_train = model(x_train)
        regualizers = model.regualizer()

        if (REGUALIZER_SCALING == 'linear'):
            r_w_scale = max(0, min(1, (
                (epoch_i - REGUALIZER_SCALING_START) /
                (REGUALIZER_SCALING_END - REGUALIZER_SCALING_START)
            )))
        elif (REGUALIZER_SCALING == 'exp'):
            r_w_scale = 1 - math.exp(-1e-5 * epoch_i)

        loss_train_criterion = criterion(y_train, t_train)
        nau_loss.append(loss_train_criterion)
        loss_train_regualizer = REGUALIZER * r_w_scale * regualizers['W'] + regualizers['g'] + REGUALIZER_Z * regualizers['z'] + REGUALIZER_OOB * regualizers['W-OOB']
        loss_train = loss_train_criterion

        # Log loss
        if VERBOSE or epoch_i % 1000 == 0:
            summary_writer.add_scalar('loss/train/critation', loss_train_criterion)
            summary_writer.add_scalar('loss/train/regualizer', loss_train_regualizer)
            summary_writer.add_scalar('loss/train/total', loss_train)
        if epoch_i % 1000 == 0:
            print('train %d: %.5f, inter: %.5f, extra: %.5f' % (epoch_i, loss_train_criterion, interpolation_error, extrapolation_error))

        # Optimize model
        if loss_train.requires_grad:
            loss_train.backward()
            optimizer.step()
        model.optimize(loss_train_criterion)

        # Log gradients if in verbose mode
        if VERBOSE and epoch_i % 1000 == 0:
            model.log_gradients()
"""

"""
=====================================================================
BASELINE
=====================================================================
"""

class Baseline(nn.Module):
    def __init__(self, hidden_dim, act_name, input_size=2, output_size=1):
        super(Baseline, self).__init__()

        self.act_name = act_name

        self.embed = nn.Embedding(256,256)
        
        self.linears = []
        self.acts = []
        
        for i in range(len(hidden_dim)):
            if i == 0:
                self.linears.append(nn.Linear(512, hidden_dim[i]))
            else:
                self.linears.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
                
            if self.act_name != 'linear':
                self.acts.append(ACTIVATIONS[self.act_name])

        self.linears.append(nn.Linear(hidden_dim[len(hidden_dim)-1], 1))
        if self.act_name != 'linear':
            self.acts.append(ACTIVATIONS[self.act_name])
        
        
        """
        self.linear1 = nn.Linear(512, hidden_dim1)
        self.act1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.act2 = nn.Sigmoid()
        self.linear3 = nn.Linear(hidden_dim2, 1)
        self.act3 = nn.Sigmoid()
        """

    def forward(self, inp):
    
        
        embedded = self.embed(inp)
        #print("Embedding:", embedded.shape)
        out = torch.flatten(embedded, start_dim=1)
        #print("Flatten:", out.shape)
        
        for i in range(len(self.linears)):
            #if self.act_name != 'linear':
                #out = self.acts[i](out)
            #print("Layer", i+1, ":", out.shape)
            out = self.linears[i](out)
            if i==0:
                out = self.act[i](out)

        #out = self.acts[0](out)
        
        return out
    
        """
        embedded = self.embed(inp)
        flat = torch.flatten(embedded, start_dim=1)
        z_1 = self.linear1(flat)
        a_1 = self.act1(z_1)
        z_2 = self.linear2(a_1)
        a_2 = self.act2(z_2)
        z_3 = self.linear3(a_2)
        a_3 = self.act3(z_3)
        return a_3
        """

    def get_model_size(self):
        # Input
        input_size = (1,1,2)
        input_bits = np.prod(input_size)*64

        # Parameters
        param_bits = 0
        param_bits += np.prod(np.array(list(self.embed.parameters())[0].size()))*32
        for lin in self.linears:
            param_bits += np.prod(np.array(list(lin.parameters())[0].size()))*32

        # Forward Backward
        forw_back_bits = 0
        ex_inp = torch.tensor([[1, 1]])
        out = self.embed(ex_inp)
        forw_back_bits += np.prod(np.array(out.size()))*32*2
        out = torch.flatten(out, start_dim=1)
        forw_back_bits += np.prod(np.array(out.size()))*32
        for i in range(len(self.linears)):
            out = self.linears[i](out)
            forw_back_bits += np.prod(np.array(out.size()))*32*2
            if self.act_name != 'linear':
                out = self.acts[i](out)
                forw_back_bits += np.prod(np.array(out.size()))*32

        return (input_bits, param_bits, forw_back_bits)
        



def dataset_gen(max_num, batch_size):

    pairs = torch.randint(max_num, (batch_size, 2))
    sums = torch.sum(pairs, dim=1) % 256
    sums = sums.reshape(-1, 1).float() 
    sums = (sums-0) / 256.
    return (pairs, sums)


EPOCHS=5000000
LEARNING_RATE=1e-3
BATCH_SIZE=128

#model = Baseline((100, 50), 'GELU')



def train_baseline(model, epochs):
    losses = []

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(epochs+1):
        mini_batch_loss = []
        for mini in range(MINI_BATCH_SIZE):

            optimizer.zero_grad()

            x_train, t_train = dataset_gen(256, BATCH_SIZE)

            y_train = model(x_train)

            loss = criterion(y_train, t_train)

            mini_batch_loss.append(loss)

            #if epoch % 1000 == 0:
            #    print("train %d: %.5f" %(epoch, loss))

            loss.backward()
            optimizer.step()

        losses.append(sum(mini_batch_loss) / len(mini_batch_loss))
        if(epoch % 500 == 0):
            print('train %d: %.5f' % (epoch, sum(mini_batch_loss)/len(mini_batch_loss)))

    return smooth(losses, 5)

"""
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
"""

"""
=====================================================================
UTIL
=====================================================================
"""

def combos(depth, dim_options):
    tmp = []
    for i in range(depth):
        tmp.append(dim_options)
    tmp = list(itertools.product(*tmp))
    return tmp

def list2string(l):
    st = ""
    for e in l:
        st += str(e)
        st += " "
    return st

def list2fn(l):
    st = ""
    for e in l:
        st += str(e)
        st += "_"
    return st

def list2csv(l):
    st = "2->512->"
    for e in l:
        st += str(e)
        st += "->"
    st += "1"
    return st

act_functions = ['linear', 'GELU', 'ReLU', 'Sigmoid','ELU', 'Tanh', 'ReLU6','LeakyReLU', 'RandReLU', 'SELU', 'CELU', 'Softplus', 'Hardshrink', 'Hardsigmoid' ,'Hardtanh', 'Hardswish', 'Tanhshrink']
num_layers = [1]
hidden_dim = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
#configs = [('ReLU6', [2]), ('Softplus', [2]), ('Hardtanh', [4]),('RandReLU', [4]),('Softplus', [4]),('Tanhshrink', [4]),('Softplus', [8]),('Tanhshrink', [8]),('Sigmoid', [16]),('Softplus', [16]),('Hardswish', [32]),('Softplus', [32])]
epoch_stop = 2000


with open('data.csv', 'w') as f:

    fields = ['Dimensions', 'Activation', 
            'Total Memory Size (MB)', 
            'Final NAU Loss', 
            'Final Baseline Loss', 
            'Final NAU Reg Loss', 
            'NAU-Baseline',
            'NAU Training Time (sec)', 
            'Baseline Training Time (sec)']


    writer = csv.DictWriter(f, fieldnames=fields, delimiter=',')
    writer.writeheader()

    for act in act_functions:
        for hid in hidden_dim:
                i = [hid, hid]
                
                print("-----------------------------------------------------------------------")
                print("Hidden Dims:", i)
                print("Activation:", act)

                print("\nTraining Baseline...")
                base_start = time.time()
                baseline = Baseline(i, act)
                base_loss = train_baseline(baseline, epoch_stop)
                base_loss = base_loss[:epoch_stop-5]
                base_end = time.time()

                print("\nTraining NAU...")
                nau_start = time.time()
                nau = NAU(i, act, unit_name=LAYER_TYPE, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, nac_oob=OOB_MODE, regualizer_shape=REGUALIZER_SHAPE, regualizer_z=REGUALIZER_Z, mnac_epsilon=MNAC_EPSILON, writer=summary_writer.every(1000).verbose(VERBOSE), nac_mul=NAC_MUL)
                nau_loss = train_nau(nau, epoch_stop)
                nau_loss = nau_loss[:epoch_stop-5]
                nau_end = time.time()

                print("\nTraining NAU Reg...")
                naureg = NAU(i, act, unit_name=LAYER_TYPE, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, nac_oob=OOB_MODE, regualizer_shape=REGUALIZER_SHAPE, regualizer_z=REGUALIZER_Z, mnac_epsilon=MNAC_EPSILON, writer=summary_writer.every(1000).verbose(VERBOSE), nac_mul=NAC_MUL)
                nau_loss_reg = train_nau_reg(naureg, epoch_stop)
                nau_loss_reg = nau_loss_reg[:epoch_stop-5]
                

                t = np.arange(epoch_stop-5)
                plt.plot(t, base_loss, 'r', label='Baseline')
                plt.plot(t, nau_loss, 'b', label='NAU')
                plt.plot(t, nau_loss_reg, 'g', label='NAU Reg')
                plt.ylim(0, 0.5)
                plt.legend(loc='upper right')
                plt.title('NAU vs. Baseline, Dim:%s, Act:%s' %(list2string(i), act))
                plt.savefig('%s%s' %(list2fn(i), act))
                plt.clf()
                
                
                """
                with open('results.txt', 'a') as f:
                    f.write('Dim:%s, Act:%s\n' %(list2string(i), act))
                    f.write('Loss after 500 epochs:\n')
                    f.write('NAU:%f, Baseline:%f, NAU - Baseline: %f\n' %(nau_loss[500], base_loss[500], nau_loss[500]-base_loss[500]))
                    f.write('Loss after 1000 epochs:\n')
                    f.write('NAU:%f, Baseline:%f, NAU - Baseline: %f\n' %(nau_loss[999], base_loss[999], nau_loss[999]-base_loss[999]))
                    f.write('\n')
                """

                writer.writerow({'Dimensions': list2csv(i), 'Activation': act, 
                'Total Memory Size (MB)': sum(baseline.get_model_size())/8000000.0, 
                'Final NAU Loss':str(nau_loss[len(nau_loss)-1].item()), 
                'Final NAU Reg Loss':str(nau_loss_reg[len(nau_loss_reg)-1].item()),
                'Final Baseline Loss':str(base_loss[len(base_loss)-1].item()), 
                'NAU-Baseline': str(nau_loss[len(nau_loss)-1].item() - base_loss[len(base_loss)-1].item()),
                'NAU Training Time (sec)':str(nau_end - nau_start), 
                'Baseline Training Time (sec)':str(base_end - base_start)})
                    
