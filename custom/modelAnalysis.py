import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import sys
sys.path.append('./')
from openstl.utils import (show_video_line, show_video_gif_multiple)
from custom.utils import load_dtparser, update_config
from openstl.utils import load_config

class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)

args = load_dtparser().parse_args()
opt = args.__dict__
# Create load_config object
config = update_config(opt, load_config("./custom/configs/yt_data.py"))
config = Config(config)

# saved model directory
saved_model_dir = './work_dirs/custom_yt_data_normalized_10_epochs_10_10_SimVP_gSTA/saved'

# make images directory under the parent of saved model directory
image_dir = os.path.join(saved_model_dir, '../images')
os.makedirs(image_dir, exist_ok=True)

# show the given frames 
inputs = np.load(saved_model_dir + '/inputs.npy')
preds = np.load(saved_model_dir + '/preds.npy')
trues = np.load(saved_model_dir + '/trues.npy')


# check the dimensions of the inputs, preds and trues
print(inputs.shape)
print(preds.shape)
print(trues.shape)


# number of input and output frames
pre_seq_length = config.n_past
aft_seq_length = config.n_future

# show the video frames for a random sample

# print(inputs.shape[0])
# Check if the number of images are greater than 10 
num_images = np.minimum(10, inputs.shape[0])

for sample_idx in range(num_images):
    # displaying input frames
    # sample_idx = np.random.randint(0, inputs.shape[0])
    # convert inputs, preds and trues to 0-255 range





    input = inputs[sample_idx]
    input = np.clip(input, 0.0, 1.0)
    input = (input * 255.0)
    input = input.astype(np.uint8)

    pred = preds[sample_idx]
    pred = np.clip(pred, 0.0, 1.0)
    pred = (preds[sample_idx] * 255.0)
    pred = pred.astype(np.uint8)

    true = trues[sample_idx]
    true = np.clip(true, 0.0, 1.0)
    true = (trues[sample_idx] * 255.0)
    true = true.astype(np.uint8)

    # print('Minimum and maximum values of inputs: ', np.min(pred), np.max(pred))

    # input = (inputs[sample_idx] - np.min(inputs[sample_idx])) / (np.max(inputs[sample_idx]) - np.min(inputs[sample_idx]))
    # pred = (preds[sample_idx] - np.min(preds[sample_idx])) / (np.max(preds[sample_idx]) - np.min(preds[sample_idx]))
    # true = (trues[sample_idx] - np.min(trues[sample_idx])) / (np.max(trues[sample_idx]) - np.min(trues[sample_idx]))
    
    show_video_line(input, ncols=pre_seq_length, vmin = 0, vmax=255, cbar=False,\
                    out_path= image_dir + f'/inputs_{sample_idx}.png',\
                    format='png', use_rgb=True)

    # displaying predicted frames
    show_video_line(pred, ncols=aft_seq_length, vmin=0, vmax=255, cbar=False,\
                    out_path= image_dir + f'/preds_{sample_idx}.png',\
                    format='png', use_rgb=True)


    # displaying ground truth frames
    show_video_line(true, ncols=aft_seq_length, vmin=0, vmax=255, cbar=False,\
                    out_path= image_dir + f'/trues_{sample_idx}.png',\
                    format='png', use_rgb=True)

    # # generate gif fil based on the input, predicted and ground truth frames
    # show_video_gif_multiple(inputs[sample_idx], trues[sample_idx], preds[sample_idx],\
    #                          out_path=image_dir + f'custom_yt_Interview_{sample_idx}.gif', use_rgb=True)
