import numpy as np
import scipy
import datetime as dt
import os
import json

from scores import GridScorer

class Ratemap:
    def __init__(self, options, res, ratemaps):
        self.options = options
        self.ratemaps = ratemaps  # (Ng, res, res)
        
        # =======================================
        # compute grid scores and sacs
        
        # DeepMind param
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        # Dehong Xu param
        # starts = [0.1] * 20
        # ends = np.linspace(0.2, 1.2, num=20)
        
        mask_parameters = zip(starts, ends.tolist())
        
        box_width = options.box_width
        box_height = options.box_height
        coord_range = ((-box_width / 2., box_width / 2.), (-box_height / 2., box_height / 2.))
        
        scorer = GridScorer(res, coord_range, mask_parameters)

        score_60, score_90, max_60_mask, max_90_mask, sacs, max_60_ind = zip(
            *[scorer.get_scores(rm) for rm in ratemaps]
        )
        # =======================================
        
        self.score_60 = score_60
        self.score_90 = score_90
        self.max_60_mask = max_60_mask
        self.max_90_mask = max_90_mask
        self.sacs = sacs
        self.max_60_ind = max_60_ind
        
        self.grid_thresh = 0.37
        
    def get_grid_scale(self):
        grid_scale_in_m = []
        
        for i in range(len(self.max_60_mask)):  # Ng
            if self.score_60[i] >= self.grid_thresh:
                mask = self.max_60_mask[i]
                scale = mask[1] * (self.options.box_width) / 2.
                grid_scale_in_m.append(scale)

        return grid_scale_in_m
    

def save_options(options):
    filename = os.path.join(options.save_dir, options.run_ID)
    if not os.path.isdir(filename):
        os.makedirs(filename, exist_ok=True)
    filename = os.path.join(filename, 'options.json')
    
    # with open(filename, 'w') as file:
    #     json.dump(options.__dict__, file)
    
    data = options.__dict__
    with open(filename, 'w') as json_file:
        json_file.write('{\n')
        for key, value in data.items():
            json_file.write(f'"{key}": {json.dumps(value)},\n')
        # Remove the last comma and add closing brace
        json_file.seek(json_file.tell() - 2, 0)  # Move back to remove the last comma
        json_file.write('\n}')  # Write the closing brace
    

def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant parameters. 
    Remaining parameters can be found in params.npy file. 
    '''
    date_time = dt.datetime.now().strftime("%m%d_%H%M%S")
    # date_time = dt.datetime.now().strftime("%m%d_%H%M")
    run_ID = options.RNN_type + '_' + date_time + '_' + options.neuron_type

    return run_ID


def get_2d_sort(x1, x2):
    """
    Reshapes x1 and x2 into square arrays, and then sorts
    them such that x1 increases downward and x2 increases
    rightward. Returns the order.
    """
    n = int(np.round(np.sqrt(len(x1))))
    total_order = x1.argsort()
    total_order = total_order.reshape(n, n)
    for i in range(n):
        row_order = x2[total_order.ravel()].reshape(n, n)[i].argsort()
        total_order[i] = total_order[i, row_order]
    total_order = total_order.ravel()
    return total_order


def dft(N,real=False,scale='sqrtn'):
    if not real:
        return scipy.linalg.dft(N,scale)
    else:
        cosines = np.cos(2*np.pi*np.arange(N//2+1)[None,:]/N*np.arange(N)[:,None])
        sines = np.sin(2*np.pi*np.arange(1,(N-1)//2+1)[None,:]/N*np.arange(N)[:,None])
        if N%2==0:
            cosines[:,-1] /= np.sqrt(2)
        F = np.concatenate((cosines,sines[:,::-1]),1)
        F[:,0] /= np.sqrt(N)
        F[:,1:] /= np.sqrt(N/2)
        return F


def load_trained_weights(model, trainer, weight_dir):
    ''' Load weights stored as a .npy file (for github)'''

    # Train for a single step to initialize weights
    trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array
    weights = np.load(weight_dir, allow_pickle=True)
    model.set_weights(weights)
    print('Loaded trained weights.')

