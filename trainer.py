import torch
import numpy as np

from visualize import save_ratemaps, save_ratemaps_rsnn, save_loss
import os
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR  #, MultiStepLR, LambdaLR, ExponentialLR


class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=True):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        
        lr = self.options.learning_rate
        
        # model params
        if options.RNN_type == 'RSNN':
            params = [
                {'params': self.model.get_params()['base_params'], 'lr': lr},
                {'params': self.model.get_params()['tau_params'], 'lr': lr * 2.},
                # {'params': self.model.get_params()['tau_params'], 'lr': lr},
                # {'params': self.model.get_params()['tau_params'], 'lr': lr * 0.5},
                {'params': self.model.get_params()['other_params'], 'lr': lr}
            ]
        else:
            params = self.model.parameters()
        
        # optimizer
        if options.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=lr)
        elif options.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(params, lr=lr, alpha=0.9, eps=1e-10, momentum=0.9)
        else:
            raise NotImplementedError

        # scheduler
        if options.scheduler == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            self.scheduler = None

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'model.pth')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))


    def train_step(self, inputs, pc_outputs, pos):
        ''' 
        Train on one batch of trajectories.
        Args:
            inputs: Batch of 2d velocity inputs with shape 
                [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape 
                [batch_size, sequence_length, 2].
        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()


    def train(self, n_epochs=100, n_steps=1000, save=True):
        ''' 
        Train model on simulated trajectories.
        Args:
            n_epochs
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''
        # Construct generator
        gen = self.trajectory_generator.get_generator()

        epoch_bar = tqdm(range(n_epochs), leave=True)
        # step_bar = tqdm(range(n_steps), leave=True)
        for epoch_idx in epoch_bar:  # range(n_epochs):
            for step_idx in range(n_steps):  # step_bar:
                self.model.train()
                
                inputs, pc_outputs, pos = next(gen)  # one batch
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)
                # step_bar.set_description(desc, refresh=True)
                
            if self.scheduler:
                self.scheduler.step()

            desc = 'Epoch: {}/{}. Step {}/{}. Loss: {}. Err: {}cm'.format(
                epoch_idx, n_epochs, step_idx, n_steps,
                np.round(loss, 2), np.round(100 * err, 2)
            )
            epoch_bar.set_description(desc, refresh=False)

            if save:
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model.pth'))
                save_loss(self.err, self.loss, self.ckpt_dir)
                
                if epoch_idx % 100 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, 'model_{}.pth'.format(epoch_idx)))
                    
                if epoch_idx % 20 == 0:    
                    self.model.eval()
                    # save_ratemaps(self.model, self.trajectory_generator, self.options, step=epoch_idx)
                    save_ratemaps_rsnn(self.model, self.trajectory_generator, self.options, step=epoch_idx)
                    
                    