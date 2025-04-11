# import numpy as np
# import tensorflow as tf
import torch

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import generate_run_ID, save_options
from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator
from model import RNN, LSTM, RSNN
from trainer import Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir',
                    default='models/LSTM/',
                    type=str, help='directory to save trained models')
# model
parser.add_argument('--RNN_type',
                    default='LSTM',
                    type=str, help='RNN, LSTM, RSNN')
parser.add_argument('--nonlinearity',
                    default='relu',  #'tanh',  #'sigmoid'
                    type=str, help='recurrent nonlinearity for glayer')
parser.add_argument('--Np',
                    default=512,
                    type=int, help='number of place cells')
parser.add_argument('--Nlstm',
                    default=512, #128,
                    type=int, help='number of LSTM hidden units')
parser.add_argument('--Nrsnn',
                    default=512, #128,
                    type=int, help='number of RSNN hidden units')
parser.add_argument('--Ng',
                    default=128, #512, #2048, #256, #4096,
                    type=int, help='number of grid cells')
# training
parser.add_argument('--n_epochs',
                    default=500,
                    type=int, help='number of training epochs')
parser.add_argument('--n_steps',
                    default=1000,
                    type=int, help='number of batches per epoch')
parser.add_argument('--batch_size',
                    default=200,
                    type=int, help='number of sample trajectories per batch')
parser.add_argument('--optimizer',
                    default='Adam', # 'RMSprop'
                    type=str, help='optimizer')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float, help='gradient descent learning rate')
parser.add_argument('--dropout_rate',
                    default=0.5,
                    type=float, help='dropout rate of glayer')
parser.add_argument('--weight_decay',
                    default=1e-5, #1e-4,
                    type=float, help='strength of weight decay on recurrent weights')
parser.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    type=str, help='device to use for training')
# data
parser.add_argument('--sequence_length',
                    default=20, #50, #100，
                    type=int, help='number of move steps in trajectory')
parser.add_argument('--place_cell_rf',
                    default=0.12, #0.2,
                    type=float, help='sigma1, width of place cell center tuning curve (m)')
parser.add_argument('--surround_scale',
                    default=2.0,
                    type=float, help='if DoS, ratio of sigma2^2 to sigma1^2')
parser.add_argument('--DoS',
                    default=True,
                    help='use difference of gaussians tuning curves')
parser.add_argument('--periodic',
                    default=False,
                    type=bool, help='trajectories with periodic boundary conditions')
parser.add_argument('--box_width',
                    default=2.2,
                    type=float, help='width of training environment')
parser.add_argument('--box_height',
                    default=2.2,
                    type=float, help='height of training environment')
# description
parser.add_argument('--desc',
                    default='LSTM',
                    type=str, help='description of the run')

options = parser.parse_args()
options.run_ID = generate_run_ID(options)
print(f'run_ID: {options.run_ID}')

# Save the options dictionary to a json file
save_options(options)


# '''
place_cells = PlaceCells(options)

if options.RNN_type == 'RNN':
    model = RNN(options, place_cells)
elif options.RNN_type == 'LSTM':
    model = LSTM(options, place_cells)
elif options.RNN_type == 'RSNN':
    model = RSNN(options, place_cells)
else:
    raise NotImplementedError   

model = model.to(options.device)

trajectory_generator = TrajectoryGenerator(options, place_cells)

trainer = Trainer(options, model, trajectory_generator)

trainer.train(n_epochs=options.n_epochs, n_steps=options.n_steps, save=True)
# '''
