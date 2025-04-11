import torch
import torch.nn as nn
import torch.nn.functional as F

from RSNN_layers.spike_neuron import b0_value
from RSNN_layers.spike_dense import (
    spike_dense_if, spike_dense_lif, spike_dense_alif, spike_dense_dexat, spike_dense_texat, 
    readout_integrator_if, readout_integrator_lif
)
from RSNN_layers.spike_rnn import (
    spike_rnn_if, spike_rnn_lif, spike_rnn_alif, spike_rnn_dexat, spike_rnn_texat
)


class RSNN(nn.Module):
    def __init__(self, options, place_cells):
        super().__init__()
        
        self.sequence_length = options.sequence_length
        self.Np = options.Np
        self.Nrsnn = options.Nrsnn  # RSNN hidden size
        self.Ng = options.Ng
        
        self.dropout_rate = options.dropout_rate
        self.weight_decay = options.weight_decay
        
        self.place_cells = place_cells
        self.device = options.device

        self.is_bias = True

        # input weights, proj p0 -> mems
        self.encoder_1 = nn.Linear(self.Np, self.Nrsnn)
        self.encoder_2 = nn.Linear(self.Np, self.Nrsnn)
        self.encoder_3 = nn.Linear(self.Np, self.Nrsnn)
        self.encoder_4 = nn.Linear(self.Np, self.Nrsnn)
        self.encoder_5 = nn.Linear(self.Np, self.Nrsnn)
        self.sigmoid = nn.Sigmoid()
        
        # ==============================================
        # RSNN layers
        self.neuron_type = options.neuron_type
        if self.neuron_type == 'if':
            self.dense_in = spike_dense_if(
                2, self.Nrsnn,
                device=self.device, bias=self.is_bias
            )
            self.rnn_1 = spike_rnn_if(
                self.Nrsnn, self.Nrsnn,
                device=self.device, bias=self.is_bias
            )
            self.rnn_2 = spike_rnn_if(
                self.Nrsnn, self.Nrsnn,
                device=self.device, bias=self.is_bias
            )
            self.rnn_3 = spike_rnn_if(
                self.Nrsnn, self.Nrsnn,
                device=self.device, bias=self.is_bias
            )
            self.dense_out = readout_integrator_if(
                self.Nrsnn, self.Nrsnn,
                device=self.device, bias=self.is_bias
            )
        elif self.neuron_type == 'lif':
            self.dense_in = spike_dense_lif(
                2, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_1 = spike_rnn_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_2 = spike_rnn_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_3 = spike_rnn_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                device=self.device, bias=self.is_bias
            )
            self.dense_out = readout_integrator_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=10, tauM_inital_std=1, 
                device=self.device, bias=self.is_bias
            )
        elif self.neuron_type == 'alif':
            self.dense_in = spike_dense_alif(
                2, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=200, tauAdp_inital_std=50, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_1 = spike_rnn_alif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=200, tauAdp_inital_std=50, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_2 = spike_rnn_alif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=200, tauAdp_inital_std=50, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_3 = spike_rnn_alif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=200, tauAdp_inital_std=50, 
                device=self.device, bias=self.is_bias
            )
            self.dense_out = readout_integrator_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=10, tauM_inital_std=1, 
                device=self.device, bias=self.is_bias
            )
        elif self.neuron_type == 'dexat':
            tauAdp_inital = [100, 200]   # [10, 20]
            tauAdp_inital_std = [20, 50]  # [2, 5]
            
            self.dense_in = spike_dense_dexat(
                2, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_1 = spike_rnn_dexat(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_2 = spike_rnn_dexat(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_3 = spike_rnn_dexat(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.dense_out = readout_integrator_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=10, tauM_inital_std=1, 
                device=self.device, bias=self.is_bias
            ) 
        elif self.neuron_type == 'texat':
            tauAdp_inital = [10, 100, 1000]
            tauAdp_inital_std = [1, 20, 100]
            
            self.dense_in = spike_dense_texat(
                2, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_1 = spike_rnn_texat(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_2 = spike_rnn_texat(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.rnn_3 = spike_rnn_texat(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=20, tauM_inital_std=5, 
                tauAdp_inital=tauAdp_inital, tauAdp_inital_std=tauAdp_inital_std, 
                device=self.device, bias=self.is_bias
            )
            self.dense_out = readout_integrator_lif(
                self.Nrsnn, self.Nrsnn,
                tauM_inital=10, tauM_inital_std=1, 
                device=self.device, bias=self.is_bias
            )
        else:
            raise NotImplementedError
        
        # RSNN layers initialization
        nn.init.xavier_uniform_(self.dense_in.dense.weight)
        nn.init.xavier_uniform_(self.rnn_1.dense.weight)
        nn.init.orthogonal_(self.rnn_1.recurrent.weight)
        nn.init.xavier_uniform_(self.rnn_2.dense.weight)
        nn.init.orthogonal_(self.rnn_2.recurrent.weight)
        nn.init.xavier_uniform_(self.rnn_3.dense.weight)
        nn.init.orthogonal_(self.rnn_3.recurrent.weight)
        nn.init.xavier_uniform_(self.dense_out.dense.weight)
        if self.is_bias:
            nn.init.constant_(self.dense_in.dense.bias, 0.)
            nn.init.constant_(self.rnn_1.dense.bias, 0.)
            nn.init.constant_(self.rnn_1.recurrent.bias, 0.)
            nn.init.constant_(self.rnn_2.dense.bias, 0.)
            nn.init.constant_(self.rnn_2.recurrent.bias, 0.)
            nn.init.constant_(self.rnn_3.dense.bias, 0.)
            nn.init.constant_(self.rnn_3.recurrent.bias, 0.)
            nn.init.constant_(self.dense_out.dense.bias, 0.)
        # ==============================================
        
        # grid activation
        # self.glayer = nn.Linear(self.Nrsnn, self.Ng, bias=False)
        
        if options.nonlinearity == 'relu':
            self.nonlin = nn.ReLU()
        elif options.nonlinearity == 'tanh':
            self.nonlin = nn.Tanh()
        elif options.nonlinearity == 'sigmoid':
            self.nonlin = nn.Sigmoid()
        else:  # 'none'
            self.nonlin = nn.Identity()
        
        # linear read-out weights
        self.decoder_1 = nn.Linear(self.Nrsnn, self.Nrsnn)
        self.decoder_2 = nn.Linear(self.Nrsnn, self.Np)
        
        self.softmax = nn.Softmax(dim=-1)

    def get_grids(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            g: Batch of grid cell activations with shape [sequence_length, batch_size, Ng].
        '''
        v, p0 = inputs
        # v: (sequence_length, batch_size, 2)
        # p0: (batch_size, Np)
        
        seq_length, batch_size, input_dim = v.size()
        
        # initialize RSNN neuron states
        self.dense_in.set_neuron_state(batch_size)
        self.rnn_1.set_neuron_state(batch_size)
        self.rnn_2.set_neuron_state(batch_size)
        self.rnn_3.set_neuron_state(batch_size)
        self.dense_out.set_neuron_state(batch_size)
        
        # proj p0 -> mems (batch_size, Nrsnn)
        p0 = p0 * 1e3
        ratio = 1.5
        # ratio = 1.75  # 1.25  #1.5  # 1.0  # 2 
        self.dense_in.mem  = ratio * b0_value * self.sigmoid(self.encoder_1(p0))
        self.rnn_1.mem     = ratio * b0_value * self.sigmoid(self.encoder_2(p0))
        self.rnn_2.mem     = ratio * b0_value * self.sigmoid(self.encoder_3(p0))
        self.rnn_3.mem     = ratio * b0_value * self.sigmoid(self.encoder_4(p0))
        self.dense_out.mem = ratio * b0_value * self.sigmoid(self.encoder_5(p0))
        
        g_mem_in = []
        g_spike_in = []
        g_mem_rnn_1 = []
        g_spike_rnn_1 = []
        g_mem_rnn_2 = []
        g_spike_rnn_2 = []
        g_mem_rnn_3 = []
        g_spike_rnn_3 = []
        g_mem_out = []
        
        rsnn_output = []
        
        for i in range(seq_length):  # one time step in the sequence
            rsnn_input = v[i]
            
            mem_in, spike_in       = self.dense_in.forward(rsnn_input)
            mem_rnn_1, spike_rnn_1 = self.rnn_1.forward(spike_in)
            mem_rnn_2, spike_rnn_2 = self.rnn_2.forward(spike_rnn_1)
            mem_rnn_3, spike_rnn_3 = self.rnn_3.forward(spike_rnn_2)
            mem_out                = self.dense_out.forward(spike_rnn_3)
            
            # non-negativity constraint
            mem_out = self.nonlin(mem_out)
            
            g_mem_in.append(mem_in)
            g_spike_in.append(spike_in)
            g_mem_rnn_1.append(mem_rnn_1)
            g_spike_rnn_1.append(spike_rnn_1)
            g_mem_rnn_2.append(mem_rnn_2)
            g_spike_rnn_2.append(spike_rnn_2)
            g_mem_rnn_3.append(mem_rnn_3)
            g_spike_rnn_3.append(spike_rnn_3)
            g_mem_out.append(mem_out)
            
            # dropout
            if self.training and self.dropout_rate > 0:
                mem_out = F.dropout(mem_out, self.dropout_rate)
                
            rsnn_output.append(mem_out)
            
        # (sequence_length, batch_size, Ng)
        g_mem_in      = torch.stack(g_mem_in, dim=0)  
        g_spike_in    = torch.stack(g_spike_in, dim=0)
        g_mem_rnn_1   = torch.stack(g_mem_rnn_1, dim=0)
        g_spike_rnn_1 = torch.stack(g_spike_rnn_1, dim=0)
        g_mem_rnn_2   = torch.stack(g_mem_rnn_2, dim=0)
        g_spike_rnn_2 = torch.stack(g_spike_rnn_2, dim=0)
        g_mem_rnn_3   = torch.stack(g_mem_rnn_3, dim=0)
        g_spike_rnn_3 = torch.stack(g_spike_rnn_3, dim=0)
        g_mem_out     = torch.stack(g_mem_out, dim=0)
        
        rsnn_output = torch.stack(rsnn_output, dim=0)
        # rsnn_output = g_mem_out
        
        return (
            rsnn_output, 
            g_mem_in, g_spike_in, 
            g_mem_rnn_1, g_spike_rnn_1, 
            g_mem_rnn_2, g_spike_rnn_2, 
            g_mem_rnn_3, g_spike_rnn_3, 
            g_mem_out,
        )

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            place_preds: Predicted place cell activations with shape [sequence_length, batch_size, Np].
        '''        
        # rsnn_output, g_mem_in, g_spike_in, g_mem_rnn_1, g_spike_rnn_1, g_mem_out = self.get_grids(inputs)
        # rsnn_output, g_mem_in, g_spike_in, g_mem_rnn_1, g_spike_rnn_1, g_mem_rnn_2, g_spike_rnn_2, g_mem_out = self.get_grids(inputs)
        (
            rsnn_output, 
            g_mem_in, g_spike_in, 
            g_mem_rnn_1, g_spike_rnn_1, 
            g_mem_rnn_2, g_spike_rnn_2, 
            g_mem_rnn_3, g_spike_rnn_3, 
            g_mem_out,
        ) = self.get_grids(inputs)
        
        place_preds = self.decoder_1(rsnn_output)
        place_preds = self.decoder_2(place_preds)
        
        # (sequence_length, batch_size, Np)
        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
            pc_outputs: Ground truth place cell activations with shape [sequence_length, batch_size, Np].
            pos: Ground truth 2d position with shape [sequence_length, batch_size, 2].
        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in m.
        '''
        y = pc_outputs
        
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        
        loss = -(y * torch.log(yhat)).sum(-1).mean()
        
        # Weight regularization
        if self.weight_decay > 0:
            # loss += self.weight_decay * (self.glayer.weight.norm(2)).sum()
            weight_loss = (
                (self.dense_in.dense.weight.norm(2)).sum() +
                (self.rnn_1.dense.weight.norm(2)).sum() +
                (self.rnn_1.recurrent.weight.norm(2)).sum() +
                (self.rnn_2.dense.weight.norm(2)).sum() +
                (self.rnn_2.recurrent.weight.norm(2)).sum() +
                (self.rnn_3.dense.weight.norm(2)).sum() +
                (self.rnn_3.recurrent.weight.norm(2)).sum() +
                (self.dense_out.dense.weight.norm(2)).sum()
            )
            weight_loss *= self.weight_decay
            loss += weight_loss

        # Compute decoding error (m)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err
    
    def parameters(self):
        base_params = self.get_params()['base_params']
        tau_params = self.get_params()['tau_params']
        other_params = self.get_params()['other_params']

        all_params = base_params + tau_params + other_params
        return all_params
    
    def get_params(self):
        if self.is_bias:
            base_params = [
                self.dense_in.dense.weight,
                self.dense_in.dense.bias,

                self.rnn_1.dense.weight,
                self.rnn_1.dense.bias,
                self.rnn_1.recurrent.weight,
                self.rnn_1.recurrent.bias,
                
                self.rnn_2.dense.weight,
                self.rnn_2.dense.bias,
                self.rnn_2.recurrent.weight,
                self.rnn_2.recurrent.bias,
                
                self.rnn_3.dense.weight,
                self.rnn_3.dense.bias,
                self.rnn_3.recurrent.weight,
                self.rnn_3.recurrent.bias,

                self.dense_out.dense.weight,
                self.dense_out.dense.bias,
            ]
        else:
            base_params = [
                self.dense_in.dense.weight,
                self.rnn_1.dense.weight,
                self.rnn_1.recurrent.weight,
                self.rnn_2.dense.weight,
                self.rnn_2.recurrent.weight,
                self.rnn_3.dense.weight,
                self.rnn_3.recurrent.weight,
                self.dense_out.dense.weight,
            ]
        
        if self.neuron_type == 'if':
            tau_params = []
        elif self.neuron_type == 'lif':
            tau_params = [
                self.dense_in.tau_m,
                # self.dense_in.tau_adp,
                
                self.rnn_1.tau_m,
                # self.rnn_1.tau_adp,
                
                self.rnn_2.tau_m,
                # self.rnn_2.tau_adp,
                
                self.rnn_3.tau_m,
                # self.rnn_3.tau_adp,
                
                self.dense_out.tau_m,
            ]
        elif self.neuron_type == 'alif':
            tau_params = [
                self.dense_in.tau_m,
                self.dense_in.tau_adp,
                
                self.rnn_1.tau_m,
                self.rnn_1.tau_adp,
                
                self.rnn_2.tau_m,
                self.rnn_2.tau_adp,
                
                self.rnn_3.tau_m,
                self.rnn_3.tau_adp,
                
                self.dense_out.tau_m,
            ]
        elif self.neuron_type == 'dexat':
            tau_params = [
                self.dense_in.tau_m,
                self.dense_in.tau_adp[0],
                self.dense_in.tau_adp[1],
                
                self.rnn_1.tau_m,
                self.rnn_1.tau_adp[0],
                self.rnn_1.tau_adp[1],
                
                self.rnn_2.tau_m,
                self.rnn_2.tau_adp[0],
                self.rnn_2.tau_adp[1],
                
                self.rnn_3.tau_m,
                self.rnn_3.tau_adp[0],
                self.rnn_3.tau_adp[1],
                
                self.dense_out.tau_m,
            ]
        elif self.neuron_type == 'texat':
            tau_params = [
                self.dense_in.tau_m,
                self.dense_in.tau_adp[0],
                self.dense_in.tau_adp[1],
                self.dense_in.tau_adp[2],
                
                self.rnn_1.tau_m,
                self.rnn_1.tau_adp[0],
                self.rnn_1.tau_adp[1],
                self.rnn_1.tau_adp[2],
                
                self.rnn_2.tau_m,
                self.rnn_2.tau_adp[0],
                self.rnn_2.tau_adp[1],
                self.rnn_2.tau_adp[2],
                
                self.rnn_3.tau_m,
                self.rnn_3.tau_adp[0],
                self.rnn_3.tau_adp[1],
                self.rnn_3.tau_adp[2],
                
                self.dense_out.tau_m,
            ]
            
        other_params = [
            self.encoder_1.weight,
            self.encoder_1.bias,
            self.encoder_2.weight,
            self.encoder_2.bias,
            self.encoder_3.weight,
            self.encoder_3.bias,
            self.encoder_4.weight,
            self.encoder_4.bias,
            self.encoder_5.weight,
            self.encoder_5.bias,
            
            # self.glayer.weight,

            self.decoder_1.weight,
            self.decoder_1.bias,
            self.decoder_2.weight,
            self.decoder_2.bias,
        ]
        
        all_params = {
            'base_params': base_params,
            'tau_params': tau_params,
            'other_params': other_params,
        }
        return all_params

