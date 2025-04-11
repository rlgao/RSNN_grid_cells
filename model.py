import torch
import torch.nn as nn
import torch.nn.functional as F

# from RSNN_layers.spike_neuron import *
# from RSNN_layers.spike_dense import *
# from RSNN_layers.spike_rnn import *


class RNN(nn.Module):
    def __init__(self, options, place_cells):
        super().__init__()
        
        self.Np = options.Np
        self.Ng = options.Ng
        
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # input weights
        self.encoder = nn.Linear(self.Np, self.Ng, bias=False)
        
        self.RNN = nn.RNN(
            input_size=2,
            hidden_size=self.Ng,
            nonlinearity=options.nonlinearity,
            bias=False
        )
        
        # linear read-out weights
        self.decoder = nn.Linear(self.Ng, self.Np, bias=False)
        
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
        
        init_state = self.encoder(p0)[None]  # (1, batch_size, Ng)
        
        g, _ = self.RNN(v, init_state)  # (sequence_length, batch_size, Ng)
        
        return g

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            place_preds: Predicted place cell activations with shape [sequence_length, batch_size, Np].
        '''
        place_preds = self.decoder(self.get_grids(inputs))
        
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
        loss += self.weight_decay * (self.RNN.weight_hh_l0 ** 2).sum()

        # Compute decoding error (m)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err


class LSTM(nn.Module):
    def __init__(self, options, place_cells):
        super().__init__()
        
        self.Np = options.Np
        self.Nlstm = options.Nlstm  # LSTM hidden size
        self.Ng = options.Ng
        
        self.sequence_length = options.sequence_length
        
        self.dropout_rate = options.dropout_rate
        self.weight_decay = options.weight_decay
        
        self.place_cells = place_cells

        # input weights
        self.encoder1 = nn.Linear(self.Np, self.Nlstm) #, bias=False)
        self.encoder2 = nn.Linear(self.Np, self.Nlstm) #, bias=False)
        
        # LSTM cell
        self.LSTM = nn.LSTMCell(input_size=2, hidden_size=self.Nlstm)
        
        # grid activation
        self.glayer = nn.Linear(self.Nlstm, self.Ng, bias=False)
        
        if options.nonlinearity == 'relu':
            self.nonlin = nn.ReLU()
        elif options.nonlinearity == 'tanh':
            self.nonlin = nn.Tanh()
        elif options.nonlinearity == 'sigmoid':
            self.nonlin = nn.Sigmoid()
        else:
            self.nonlin = nn.Identity()
        
        # linear read-out weights
        self.decoder = nn.Linear(self.Ng, self.Np) #, bias=False)
        
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
        
        hx = self.encoder1(p0)  # (batch_size, Nlstm)
        cx = self.encoder2(p0)  # (batch_size, Nlstm)
        
        g = []
        for i in range(v.size()[0]):  # sequence_length
            # one time step in the sequence
            hx, cx = self.LSTM(v[i], (hx, cx))
            
            g_step = self.glayer(hx)  # (batch_size, Ng)
            
            # non-negativity constraint
            g_step = self.nonlin(g_step)
            
            if self.training and self.dropout_rate > 0:
                g_step = F.dropout(g_step, self.dropout_rate)
                
            g.append(g_step)
            
        g = torch.stack(g, dim=0)  # (sequence_length, batch_size, Ng)
        
        return g

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [sequence_length, batch_size, 2].
        Returns: 
            place_preds: Predicted place cell activations with shape [sequence_length, batch_size, Np].
        '''
        place_preds = self.decoder(self.get_grids(inputs))  # (sequence_length, batch_size, Np)
        
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
        loss += self.weight_decay * (self.glayer.weight.norm(2)).sum()

        # Compute decoding error (m)
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err

