import torch
import torch.nn as nn

from .spike_neuron import (
    b0_value, 
    if_neuron_dynamics, lif_neuron_dynamics, alif_neuron_dynamics, dexat_neuron_dynamics, texat_neuron_dynamics,
    output_neuron_dynamics,
    multi_normal_initilization
)


class spike_dense_if(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
    
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim) * self.b0).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.spike = (torch.zeros(batch_size, self.output_dim)).to(self.device)

    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())  # different from spike_rnn

        self.mem, self.spike = if_neuron_dynamics(
            inputs=d_input, mem=self.mem, spike=self.spike,
            device=self.device
        )
        
        return self.mem, self.spike
    

class spike_dense_lif(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        tauM_inital=20, tauM_inital_std=5,
        tau_initializer='normal', 
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m, tauM_inital, tauM_inital_std)
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m, tauM_inital, tauM_inital_std)
        
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim) * self.b0).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.spike = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())  # different from spike_rnn

        self.mem, self.spike = lif_neuron_dynamics(
            inputs=d_input, mem=self.mem, spike=self.spike, 
            tau_m=self.tau_m,
            device=self.device
        )
        
        return self.mem, self.spike
    
    
class spike_dense_alif(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        tauM_inital=20, tauM_inital_std=5,
        tauAdp_inital=200, tauAdp_inital_std=5,
        tau_initializer='normal', 
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.b0 = b0_value
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m, tauM_inital, tauM_inital_std)
            nn.init.normal_(self.tau_adp, tauAdp_inital, tauAdp_inital_std)
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m, tauM_inital, tauM_inital_std)
            self.tau_adp = multi_normal_initilization(self.tau_adp, tauAdp_inital, tauAdp_inital_std)
    
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim) * self.b0).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.spike = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.b = (torch.ones(batch_size, self.output_dim) * self.b0).to(self.device)
        # self.b = (torch.zeros(batch_size, self.output_dim)).to(self.device)
    
    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())  # different from spike_rnn

        self.mem, self.spike, theta, self.b = alif_neuron_dynamics(
            inputs=d_input, mem=self.mem, spike=self.spike, 
            tau_m=self.tau_m, 
            tau_adp=self.tau_adp, b=self.b, 
            device=self.device
        )
        
        return self.mem, self.spike


class spike_dense_dexat(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        tauM_inital=20, tauM_inital_std=5,
        tauAdp_inital=[200, 200], tauAdp_inital_std=[5, 5],
        tau_initializer='normal', 
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.b0 = b0_value
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = [nn.Parameter(torch.Tensor(self.output_dim)),
                        nn.Parameter(torch.Tensor(self.output_dim))]
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m, tauM_inital, tauM_inital_std)
            nn.init.normal_(self.tau_adp[0], tauAdp_inital[0], tauAdp_inital_std[0])
            nn.init.normal_(self.tau_adp[1], tauAdp_inital[1], tauAdp_inital_std[1])
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m, tauM_inital, tauM_inital_std)
            self.tau_adp[0] = multi_normal_initilization(self.tau_adp[0], tauAdp_inital[0], tauAdp_inital_std[0])
            self.tau_adp[1] = multi_normal_initilization(self.tau_adp[1], tauAdp_inital[1], tauAdp_inital_std[1])
    
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim) * self.b0).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.spike = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.b = [(torch.ones(batch_size, self.output_dim) * self.b0).to(self.device),
                  (torch.ones(batch_size, self.output_dim) * self.b0).to(self.device)]
        # self.b = (torch.zeros(batch_size, self.output_dim)).to(self.device)
    
    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())  # different from spike_rnn

        self.mem, self.spike, theta, self.b = dexat_neuron_dynamics(
            inputs=d_input, mem=self.mem, spike=self.spike, 
            tau_m=self.tau_m, 
            tau_adp=self.tau_adp, b=self.b, 
            device=self.device
        )
        
        return self.mem, self.spike


class spike_dense_texat(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        tauM_inital=20, tauM_inital_std=5,
        tauAdp_inital=[200, 200, 200], tauAdp_inital_std=[5, 5, 5],
        tau_initializer='normal', 
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.b0 = b0_value
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        
        self.tau_adp = [nn.Parameter(torch.Tensor(self.output_dim)),
                        nn.Parameter(torch.Tensor(self.output_dim)),
                        nn.Parameter(torch.Tensor(self.output_dim))]
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m, tauM_inital, tauM_inital_std)
            nn.init.normal_(self.tau_adp[0], tauAdp_inital[0], tauAdp_inital_std[0])
            nn.init.normal_(self.tau_adp[1], tauAdp_inital[1], tauAdp_inital_std[1])
            nn.init.normal_(self.tau_adp[2], tauAdp_inital[2], tauAdp_inital_std[2])
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m, tauM_inital, tauM_inital_std)
            self.tau_adp[0] = multi_normal_initilization(self.tau_adp[0], tauAdp_inital[0], tauAdp_inital_std[0])
            self.tau_adp[1] = multi_normal_initilization(self.tau_adp[1], tauAdp_inital[1], tauAdp_inital_std[1])
            self.tau_adp[2] = multi_normal_initilization(self.tau_adp[2], tauAdp_inital[2], tauAdp_inital_std[2])
        
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim) * self.b0).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.spike = (torch.zeros(batch_size, self.output_dim)).to(self.device)
        
        self.b = [(torch.ones(batch_size, self.output_dim) * self.b0).to(self.device),
                  (torch.ones(batch_size, self.output_dim) * self.b0).to(self.device),
                  (torch.ones(batch_size, self.output_dim) * self.b0).to(self.device)]
        # self.b = (torch.zeros(batch_size, self.output_dim)).to(self.device)
    
    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())  # different from spike_rnn

        self.mem, self.spike, theta, self.b = texat_neuron_dynamics(
            inputs=d_input, mem=self.mem, spike=self.spike, 
            tau_m=self.tau_m, 
            tau_adp=self.tau_adp, b=self.b, 
            device=self.device
        )
        
        return self.mem, self.spike
    

class readout_integrator_if(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
    
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim)).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)

    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())

        self.mem = output_neuron_dynamics(
            inputs=d_input, mem=self.mem, tau_m=None,
            type='if',
            device=self.device
        )
        
        return self.mem


class readout_integrator_lif(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        tauM_inital=20, tauM_inital_std=5, 
        tau_initializer='normal',
        device='cpu', bias=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m, tauM_inital, tauM_inital_std)
                
    def set_neuron_state(self, batch_size):
        # self.mem = (torch.rand(batch_size, self.output_dim)).to(self.device)
        self.mem = (torch.zeros(batch_size, self.output_dim)).to(self.device)

    def forward(self, input_spike):
        d_input = self.dense(input_spike.float())

        self.mem = output_neuron_dynamics(
            inputs=d_input, mem=self.mem, tau_m=self.tau_m,
            type='lif',
            device=self.device
        )
        
        return self.mem

