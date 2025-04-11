import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# b0_value = 1.6
b0_value = 0.1  # 0.01
# b0_value = 0.1

R_m = 1.0

def gaussian(x, mu, sigma):
    return torch.exp(-((x - mu)**2) / (2 * sigma**2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


# define approximate firing function
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential - threshold
        ctx.save_for_backward(input)
        dt = 1.0  # 0.02
        return input.gt(0).float() / dt

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # temp = abs(input) < lens
        scale = 6.0   # s in paper
        hight = 0.15  # h in paper
        gamma = 0.5
        lens = 0.5
        
        surrogate_type = 'MG'  # surrogate gradient type

        if surrogate_type == 'G':
            temp = torch.exp(-(input**2) / (2 * lens**2)) / torch.sqrt(2 * torch.tensor(math.pi)) / lens
        elif surrogate_type == 'MG':
            temp = gaussian(input, mu=0.0, sigma=lens) * (1.0 + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrogate_type == 'linear':
            temp = F.relu(1 - input.abs())
        elif surrogate_type == 'slayer':
            temp = torch.exp(-5 * input.abs())

        dt = 1.0  # 0.02
        return grad_input * temp.float() * gamma / dt

act_fun_adp = ActFun_adp.apply


def if_neuron_dynamics(
    inputs, mem, spike,
    dt=1.0, device=None
):
    B = b0_value
    mem = mem + R_m * inputs - B * spike * dt  # membrane potential
    inputs_ = mem - B

    # spike activation function
    spike = act_fun_adp(inputs_)
    # continuous ReLU
    # spike = F.relu(inputs_)

    return mem, spike


def lif_neuron_dynamics(
    inputs, mem, spike, 
    tau_m,
    dt=1.0, device=None
):
    alpha = torch.exp(-1. * dt / tau_m).to(device)

    B = b0_value
    mem = alpha * mem + (1. - alpha) * R_m * inputs - B * spike * dt  # membrane potential
    inputs_ = mem - B

    # spike activation function
    spike = act_fun_adp(inputs_)
    # continuous ReLU
    # spike = F.relu(inputs_)

    return mem, spike


def alif_neuron_dynamics(
    inputs, mem, spike, 
    tau_m, 
    tau_adp, b, 
    dt=1.0, device=None
):
    """
    This function updates the membrane potential and adaptation variable of a spiking neural network.
    
    Inputs:
        inputs: the input spikes to the neuron
        mem: the current membrane potential of the neuron
        spike: the current adaptation variable of the neuron
        tau_adp: the time constant for the adaptation variable
        b: a value used in the adaptation variable update equation
        tau_m: the time constant for the membrane potential
        dt: the time step used in the simulation
        is_adapt: a boolean variable indicating whether or not to use the adaptation variable
        device: a variable indicating which device (e.g. CPU or GPU) to use for the computation

    Outputs:
        mem: the updated membrane potential
        spike: the updated adaptation variable
        B: a value used in the adaptation variable update equation
        b: the updated value of the adaptation variable

    The function first computes the exponential decay factors alpha and rho using the time constants tau_m and tau_adp, respectively.
    It then checks whether the is_adapt variable is True or False to determine the value of beta.
    The adaptation variable b is then updated using the exponential decay rule, and B is computed using the value of beta and the initial value b0_value.
    The function then updates the membrane potential mem using the input spikes, B, and the decay factor alpha, and computes the inputs_ variable as the difference between mem and B.
    Finally, the adaptation variable spike is updated using the activation function defined in the act_fun_adp() function, and the updated values of mem, spike, B, and b are returned.
    """
    
    # beta = 0.184
    beta = 0.18  # 1.8
    # beta = 0.2
    
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    rho = torch.exp(-1. * dt / tau_adp).to(device)

    b = rho * b + (1. - rho) * spike  # eta in paper
    B = b0_value + beta * b  # threshold, theta in paper

    mem = alpha * mem + (1. - alpha) * R_m * inputs - B * spike * dt  # membrane potential, u in paper
    inputs_ = mem - B

    # spike activation function
    spike = act_fun_adp(inputs_)
    # continuous ReLU
    # spike = F.relu(inputs_)

    return mem, spike, B, b


def dexat_neuron_dynamics(
    inputs, mem, spike, 
    tau_m, 
    tau_adp, b,  # 2-D
    dt=1.0, device=None
):
    beta = [0.18, 0.18]
    
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    
    # rho = torch.exp(-1. * dt / tau_adp).to(device)
    rho = [torch.exp(-1. * dt / tau_adp[0]).to(device), 
           torch.exp(-1. * dt / tau_adp[1]).to(device)]

    # b = rho * b + (1. - rho) * spike  # eta in paper
    b[0] = rho[0] * b[0] + (1. - rho[0]) * spike
    b[1] = rho[1] * b[1] + (1. - rho[1]) * spike
    
    # B = b0_value + beta * b  # threshold, theta in paper
    B = b0_value + beta[0] * b[0] + beta[1] * b[1]

    mem = alpha * mem + (1. - alpha) * R_m * inputs - B * spike * dt  # membrane potential, u in paper
    inputs_ = mem - B

    # spike activation function
    spike = act_fun_adp(inputs_)
    # continuous ReLU
    # spike = F.relu(inputs_)

    return mem, spike, B, b


def texat_neuron_dynamics(
    inputs, mem, spike, 
    tau_m, 
    tau_adp, b,  # 3-D
    dt=1.0, device=None
):
    beta = [0.05, 0.1, 0.25]
    
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    
    # rho = torch.exp(-1. * dt / tau_adp).to(device)
    rho = [torch.exp(-1. * dt / tau_adp[0]).to(device), 
           torch.exp(-1. * dt / tau_adp[1]).to(device), 
           torch.exp(-1. * dt / tau_adp[2]).to(device)]

    # b = rho * b + (1. - rho) * spike  # eta in paper
    b[0] = rho[0] * b[0] + (1. - rho[0]) * spike
    b[1] = rho[1] * b[1] + (1. - rho[1]) * spike
    b[2] = rho[2] * b[2] + (1. - rho[2]) * spike
    
    # B = b0_value + beta * b  # threshold, theta in paper
    B = b0_value + beta[0] * b[0] + beta[1] * b[1] + beta[2] * b[2]

    mem = alpha * mem + (1. - alpha) * R_m * inputs - B * spike * dt  # membrane potential, u in paper
    inputs_ = mem - B

    # spike activation function
    spike = act_fun_adp(inputs_)
    # continuous ReLU
    # spike = F.relu(inputs_)

    return mem, spike, B, b


def output_neuron_dynamics(
    inputs, mem, tau_m, 
    type='lif',
    dt=1.0, device=None
):
    """
    The read out neuron is a (leaky) integrator without spikes
    """
    
    if type == 'if':
        mem = mem + R_m * inputs
    elif type == 'lif':
        alpha = torch.exp(-1. * dt / tau_m).to(device)
        mem = alpha * mem + (1. - alpha) * R_m * inputs

    return mem


def multi_normal_initilization(param, means=[10,200], stds=[5,20]):
    shape_list = param.shape
    if len(shape_list) == 1:
        num_total = shape_list[0]
    elif len(shape_list) == 2:
        num_total = shape_list[0] * shape_list[1]

    num_per_group = int(num_total / len(means))
    # if num_total%len(means) != 0: 
    num_last_group = num_total % len(means)
    
    a = []
    for i in range(len(means)):
        a = a + np.random.normal(means[i],stds[i], size=num_per_group).tolist()
        if i == len(means):
            a = a + np.random.normal(means[i],stds[i],size=num_per_group+num_last_group).tolist()
    p = np.array(a).reshape(shape_list)
    
    with torch.no_grad():
        param.copy_(torch.from_numpy(p).float())
        
    return param
