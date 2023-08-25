import warnings

warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from openstl.utils import load_config, show_video_line, create_parser, setup_multi_processes, get_dist_info
from openstl.api import BaseExperiment
from mb_multifile_debug import MB
from custom.utils import load_dtparser, update_config, sequence_input, normalize_data
from openstl.utils import update_config as upd


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)


try:
    args = load_dtparser().parse_args()
    opt = args.__dict__

    # Create load_config object
    print(">" * 70)
    config = update_config(opt, load_config("./custom/configs/yt_data.py"))
    config = Config(config)
    config.dtype = torch.cuda.FloatTensor

    # Load testing data
    test_data = MB(
        config,
        task='test',
        data_root=config.test_root,
        gs_root=config.test_root,
        audio_root=config.test_root,
    )

    # Creating dataloader for non-distributed training
    train_sampler = None
    val_sampler = None
    test_sampler = None
    config.data_threads = 2

    # Define the dataloaders using the test loaders as the train and val loaders are not used
    test_loader = DataLoader(
        test_data,
        num_workers=config.data_threads,
        batch_size=config.val_batch_size,
        sampler=test_sampler,
        shuffle=(test_sampler is None),
        pin_memory=True,
    )

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

    # set test parameter to be True
    model_config['test'] = True
    model_config['metrics'] = ['mae', 'mse', 'ssim', 'psnr']

    # set multi-process settings
    # setup_multi_processes(model_config)

    # create the experiment object
    exp = BaseExperiment(model_args)
    exp.init_experiment(dataloaders=(test_loader, test_loader, test_loader))


    print(">" * 35, " Testing ", "<" * 35)
    mse = exp.test()

    print(">" * 35, " Testing finished ", "<" * 35)
    # rank, _ = get_dist_info()
    # try:
    #     import nni
    #     has_nni = True
    # except ImportError:
    #     has_nni = False

    # if rank == 0 and has_nni:
    #         nni.report_final_result(mse)
except Exception as e:
    print(">" * 35, " Testing failed with error", "<" * 35)
    print(e)
