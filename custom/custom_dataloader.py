import warnings

warnings.filterwarnings("ignore")

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from openstl.utils import load_config, show_video_line, create_parser, setup_multi_processes
from openstl.api import BaseExperiment
from mb_multifile_debug import MB
from custom.utils import load_dtparser, update_config, sequence_input, normalize_data
from openstl.utils import update_config as upd


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)


# def get_batch(dtype, dataloader):
#     while True:
#         for (
#             image_sequence,
#             image_small_sequence,
#             gs_sequence,
#             audio_sequence,
#         ) in dataloader:
#             print("image_sequence: ", image_sequence.shape)
#             batch = normalize_data(dtype, image_sequence)
#             batch_small = normalize_data(dtype, image_small_sequence)
#             gs_batch = normalize_data(dtype, gs_sequence)
#             ad_batch = sequence_input(audio_sequence.transpose_(0, 1), dtype)
#             yield batch, batch_small, gs_batch, ad_batch


args = load_dtparser().parse_args()
opt = args.__dict__

# Create load_config object
print(">" * 70)
config = update_config(opt, load_config("./custom/configs/yt_data.py"))
config = Config(config)
config.dtype = torch.cuda.FloatTensor

# Load training and testing data
train_data = MB(
    config,
    train=True,
    data_root=config.train_root,
    gs_root=config.train_gs,
    audio_root=config.train_root,
)
test_data = MB(
    config,
    train=False,
    data_root=config.val_root,
    gs_root=config.val_gs,
    audio_root=config.val_root,
)

# Creating dataloader for non-distributed training
train_sampler = None
test_sampler = None
config.data_threads = 2


# Define the dataloaders
train_loader = DataLoader(
    train_data,
    num_workers=config.data_threads,
    batch_size=config.batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    pin_memory=True,
)

test_loader = DataLoader(
    test_data,
    num_workers=config.data_threads,
    batch_size=config.val_batch_size,
    sampler=test_sampler,
    shuffle=(test_sampler is None),
    pin_memory=True,
)

val_loader = DataLoader(
    test_data,
    num_workers=config.data_threads,
    batch_size=config.val_batch_size,
    sampler=test_sampler,
    shuffle=(test_sampler is None),
    pin_memory=True,
)

# print(type(train_loader))

# # Generate a batch of data for analysis
# for X, Y in train_loader:
#     # print(data.shape)
#     condition_frames = X
#     future_frames = Y
#     print("previous_frames: ", condition_frames.shape)
#     # show_video_line(
#     #     condition_frames[0],
#     #     ncols=config.n_past,
#     #     vmax=0.6,
#     #     cbar=False,
#     #     out_path="condition_frames.png",
#     #     format="png",
#     # )
#     print("future_frames: ", future_frames.shape)
#     # yield batch, batch_small, gs_batch, ad_batch

model_args = create_parser().parse_args()
model_config = model_args.__dict__

custom_training_config = {
    'batch_size': config.batch_size,
    'val_batch_size': config.val_batch_size,
    'in_shape': (config.n_past, 3, config.image_width, config.image_height),
    'pre_seq_length': config.n_past,
    'aft_seq_length': config.n_future,
    'total_length': config.n_past + config.n_future,
}

# update the model config with the custom training config
model_config = update_config(model_config, load_config("./custom/configs/SimVP_gSTA.py"))
# update the remaining model config with the custom training config
model_config = update_config(model_config, custom_training_config)


# set multi-process settings
# setup_multi_processes(model_config)

# create the experiment object
exp = BaseExperiment(model_args, dataloaders=(train_loader, val_loader, test_loader))

# clear cuda cache
torch.cuda.empty_cache()

# run the experiment
print(">" * 35, " Training ", "<" * 35)
exp.train()

print(">" * 35, " Testing ", "<" * 35)
exp.test()