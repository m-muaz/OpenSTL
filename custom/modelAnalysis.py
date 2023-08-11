import warnings
warnings.filterwarnings("ignore")
import numpy as np 
from openstl.utils import (show_video_line, show_video_gif_multiple)
from openstl.datasets import dataset_parameters

# show the given frames 
inputs = np.load('../work_dirs/mmnist_simvp_gsta_200_epochs/saved/inputs.npy')
preds = np.load('../work_dirs/mmnist_simvp_gsta_200_epochs/saved/preds.npy')
trues = np.load('../work_dirs/mmnist_simvp_gsta_200_epochs/saved/trues.npy')


# check the dimensions of the inputs, preds and trues
print(inputs.shape)
print(preds.shape)
print(trues.shape)


# number of input and output frames
pre_seq_length = dataset_parameters["mmnist"]["pre_seq_length"]
aft_seq_length = dataset_parameters["mmnist"]["aft_seq_length"]

# show the video frames for a random sample

# displaying input frames
sample_idx = np.random.randint(0, inputs.shape[0])
show_video_line(inputs[sample_idx], ncols=pre_seq_length, vmax=0.6, cbar=False, out_path='inputs_200.png',\
                format='png', use_rgb=True)

# displaying predicted frames
show_video_line(preds[sample_idx], ncols=pre_seq_length, vmax=0.6, cbar=False, out_path='preds_200.png',\
                format='png', use_rgb=True)


# displaying ground truth frames
show_video_line(trues[sample_idx], ncols=pre_seq_length, vmax=0.6, cbar=False, out_path='trues_200.png',\
                format='png', use_rgb=True)

# generate gif fil based on the input, predicted and ground truth frames
show_video_gif_multiple(inputs[sample_idx], trues[sample_idx], preds[sample_idx], out_path='mnist_simvp_epochs_200.gif', use_rgb=True)
