import numpy as np
from matplotlib import pyplot as plt

import scipy
import scipy.stats
from imageio import imsave
import cv2
import os

from utils import Ratemap
from scores import GridScorer


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
            
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row : column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size - 1:
            # Add spacer
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(ratemaps, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in ratemaps[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, ratemaps.shape[-1])
    return rm_fig


def compute_ratemaps(model, trajectory_generator, options, res=20, n_avg=None, idxs=None):
    '''Compute spatial firing fields'''
    Ng = options.Ng

    if not n_avg:
        # n_avg = 1000 // options.sequence_length
        n_avg = 50

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    # g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    # pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    ratemaps = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        
        #####################################################
        g_batch = model.get_grids(inputs).detach().cpu().numpy()
        #####################################################
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        # g[index] = g_batch
        # pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size * options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >= 0 and x < res and y >= 0 and y < res:
                counts[int(x), int(y)] += 1
                ratemaps[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                ratemaps[:, x, y] /= counts[x, y]
                
    # g = g.reshape([-1, Ng])
    # pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # ratemaps = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    
    # ratemaps: (Ng, res, res)
    # return ratemaps, g, pos
    return ratemaps


def compute_ratemaps_rsnn(model, trajectory_generator, options, res=20, n_avg=None, idxs=None):
    '''Compute spatial firing fields'''
    Ng = options.Ng

    if not n_avg:
        # n_avg = 1000 // options.sequence_length
        n_avg = 50

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    ratemaps = {
        'mem_in'      : np.zeros([Ng, res, res]), 
        'spike_in'    : np.zeros([Ng, res, res]), 
        'mem_rnn_1'   : np.zeros([Ng, res, res]), 
        'spike_rnn_1' : np.zeros([Ng, res, res]),  
        'mem_rnn_2'   : np.zeros([Ng, res, res]), 
        'spike_rnn_2' : np.zeros([Ng, res, res]),  
        'mem_rnn_3'   : np.zeros([Ng, res, res]), 
        'spike_rnn_3' : np.zeros([Ng, res, res]), 
        'mem_out'     : np.zeros([Ng, res, res]), 
    }
    counts = np.zeros([res, res])

    for index in range(n_avg):
        inputs, pos_batch, _ = trajectory_generator.get_test_batch()
        
        #####################################################
        # g_batch = model.get_grids(inputs).detach().cpu().numpy()
        (
            rsnn_output, 
            g_mem_in, g_spike_in, 
            g_mem_rnn_1, g_spike_rnn_1, 
            g_mem_rnn_2, g_spike_rnn_2, 
            g_mem_rnn_3, g_spike_rnn_3, 
            g_mem_out,
        ) = model.get_grids(inputs)
        #####################################################
        
        g_mem_in      = g_mem_in.detach().cpu().numpy()
        g_spike_in    = g_spike_in.detach().cpu().numpy()
        g_mem_rnn_1   = g_mem_rnn_1.detach().cpu().numpy()
        g_spike_rnn_1 = g_spike_rnn_1.detach().cpu().numpy()
        g_mem_rnn_2   = g_mem_rnn_2.detach().cpu().numpy()
        g_spike_rnn_2 = g_spike_rnn_2.detach().cpu().numpy()
        g_mem_rnn_3   = g_mem_rnn_3.detach().cpu().numpy()
        g_spike_rnn_3 = g_spike_rnn_3.detach().cpu().numpy()
        g_mem_out     = g_mem_out.detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        
        g_mem_in      = g_mem_in[:,:,idxs].reshape(-1, Ng)
        g_spike_in    = g_spike_in[:,:,idxs].reshape(-1, Ng)
        g_mem_rnn_1   = g_mem_rnn_1[:,:,idxs].reshape(-1, Ng)
        g_spike_rnn_1 = g_spike_rnn_1[:,:,idxs].reshape(-1, Ng)
        g_mem_rnn_2   = g_mem_rnn_2[:,:,idxs].reshape(-1, Ng)
        g_spike_rnn_2 = g_spike_rnn_2[:,:,idxs].reshape(-1, Ng)
        g_mem_rnn_3   = g_mem_rnn_3[:,:,idxs].reshape(-1, Ng)
        g_spike_rnn_3 = g_spike_rnn_3[:,:,idxs].reshape(-1, Ng)
        g_mem_out     = g_mem_out[:,:,idxs].reshape(-1, Ng)

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size * options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >= 0 and x < res and y >= 0 and y < res:
                counts[int(x), int(y)] += 1
                ratemaps['mem_in'][:, int(x), int(y)]      += g_mem_in[i, :]
                ratemaps['spike_in'][:, int(x), int(y)]    += g_spike_in[i, :]
                ratemaps['mem_rnn_1'][:, int(x), int(y)]   += g_mem_rnn_1[i, :]
                ratemaps['spike_rnn_1'][:, int(x), int(y)] += g_spike_rnn_1[i, :]
                ratemaps['mem_rnn_2'][:, int(x), int(y)]   += g_mem_rnn_2[i, :]
                ratemaps['spike_rnn_2'][:, int(x), int(y)] += g_spike_rnn_2[i, :]
                ratemaps['mem_rnn_3'][:, int(x), int(y)]   += g_mem_rnn_3[i, :]
                ratemaps['spike_rnn_3'][:, int(x), int(y)] += g_spike_rnn_3[i, :]
                ratemaps['mem_out'][:, int(x), int(y)]     += g_mem_out[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                for idx, (key, rm) in enumerate(ratemaps.items()):
                    rm[:, x, y] /= counts[x, y]

    return ratemaps


def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None):
    if not n_avg:
        # n_avg = 1000 // options.sequence_length
        n_avg = 50
        
    ratemaps, g, pos = compute_ratemaps(model, trajectory_generator, options, res=res, n_avg=n_avg)
    # ratemaps: (Ng, res, res)
    
    # rm_fig = plot_ratemaps(ratemaps, n_plots=len(ratemaps))
    # imdir = options.save_dir + "/" + options.run_ID
    # imsave(imdir + "/" + str(step) + ".png", rm_fig)
    
    fig_rm_sac = plot_ratemaps_and_sacs(ratemaps, options, res=res)
    fig_rm_sac.savefig(os.path.join(options.save_dir, options.run_ID, (str(step) + '.png')))
    plt.close(fig_rm_sac)


def save_ratemaps_rsnn(model, trajectory_generator, options, step, res=20, n_avg=None):
    if not n_avg:
        # n_avg = 1000 // options.sequence_length
        n_avg = 50
        
    ratemaps = compute_ratemaps_rsnn(model, trajectory_generator, options, res=res, n_avg=n_avg)

    rate_maps = {}
    for idx, (key, rm) in enumerate(ratemaps.items()):
        rate_maps[key] = Ratemap(options=options, res=res, ratemaps=rm)
        
    # for i in range(len(rate_maps)):
    for idx, (key, rate_map) in enumerate(rate_maps.items()):
        fig_rm_sac = plot_ratemaps_and_sacs(rate_map, title=key, res=res, n_plot=options.Ng)
        fig_name = '_' + str(idx) + '_' + key + '.png'
        fig_rm_sac.savefig(os.path.join(options.save_dir, options.run_ID, (str(step) + fig_name)))
        plt.close(fig_rm_sac)
    

def plot_ratemaps_and_sacs(
    rate_map, 
    title=None,
    res=20, 
    n_plot=128, 
    n_cols=16, 
    sort_by_score_60=True,
    selected_idxs=None,
):
    ratemaps = rate_map.ratemaps
    score_60 = rate_map.score_60
    sacs = rate_map.sacs
    max_60_mask = rate_map.max_60_mask
    
    # plot scored ratemaps and sacs
    n_plot = n_plot if n_plot < 256 else 256
    n_rows = n_plot // n_cols

    if sort_by_score_60:  # from higher to lower
        ordering = np.argsort(-np.array(score_60))
    else:
        ordering = range(n_plot)

    fig, axes = plt.subplots(
        nrows=n_rows*2,
        ncols=n_cols,
        figsize=[n_cols, n_rows*2]
    )
    if title is not None:
        fig.suptitle(title)

    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax_rm = axes[row][col]
            ax_sac = axes[n_rows + row][col]
            
            if selected_idxs is not None:
                idx = ordering[selected_idxs[i]]
            else:
                idx = ordering[i]
            
            s60 = score_60[idx]
            rm = ratemaps[idx]
            sac = sacs[idx]
            mask = max_60_mask[idx]
            
            # plot one scored ratemap
            ax_rm.imshow(rm, interpolation='gaussian', cmap='jet')
            
            # plot one scored sac
            ax_sac.imshow(sac, interpolation='gaussian', cmap='jet')
            
            # plot mask circle in sac
            if s60 >= rate_map.grid_thresh:
                center = res - 1
                ax_sac.add_artist(
                    plt.Circle(
                        (center, center),
                        mask[1] * res,  # max_60_mask[1] * res,
                        # lw=bump_size,
                        fill=False,
                        edgecolor='r',
                    )
                )

            # ax_title = '%d (%.2f)' % (idx, s60)
            if selected_idxs is not None:
                ax_title = '%.2f' % s60
            else:
                ax_title = '%d (%.2f)' % (i, s60)
                
            ax_rm.set_title(ax_title, fontsize=8)
            # ax_sac.set_title(ax_title, fontsize=8)
            ax_rm.axis('off')
            ax_sac.axis('off')
            i += 1
            
    fig.tight_layout()
    return fig    


def save_loss(err, loss, ckpt_dir):
    err = np.array(err)
    loss = np.array(loss)
    np.save(os.path.join(ckpt_dir, 'err.npy'), err)
    np.save(os.path.join(ckpt_dir, 'loss.npy'), loss)

    plt.figure(figsize=(12,6))
    
    plt.subplot(121)
    plt.plot(err, c='blue', alpha=0.2)
    plt.plot(smooth(err, weight=0.95), c='blue')  # smoothed
    plt.title('Decoding error (m)')
    plt.xlabel('Training step')
    plt.grid(True)
    plt.ylim([0, 1.1])
    plt.yticks([i * 0.1 for i in range(12)])

    plt.subplot(122)
    plt.plot(loss, c='blue', alpha=0.2)
    plt.plot(smooth(loss, weight=0.95), c='blue')  # smoothed
    plt.title('Loss')
    plt.xlabel('Training step')
    plt.grid(True)
    
    # plt.show()
    plt.savefig(os.path.join(ckpt_dir, 'loss.png'))
    plt.close()


def smooth(scalars, weight=0.8):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1.0 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

