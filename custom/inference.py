import warnings

warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from openstl.utils import load_config, show_video_line, create_parser, setup_multi_processes, get_dist_info
from openstl.api import BaseExperiment
from mb_multifile_debug import MB
from custom.utils import load_dtparser, update_config, sequence_input, normalize_data
from openstl.utils import update_config as upd

from openstl.utils import (show_video_line, show_video_gif_multiple)


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)


try:
    args = load_dtparser().parse_known_args()[0]
    opt = args.__dict__

    # Create load_config object
    print(">" * 70)
    config = update_config(opt, load_config("./custom/configs/yt_data.py"))
    config = Config(config)
    config.dtype = torch.cuda.FloatTensor

    model_args = create_parser().parse_args()
    model_config = model_args.__dict__
    
    # update the model config with the custom training config
    model_config = update_config(model_config, load_config("./custom/configs/SimVP_gSTA.py"))
    
    # set test parameter to be True
    model_config['test'] = True
    model_config['metrics'] = ['mae', 'mse', 'ssim', 'psnr']
    
    custom_training_config = {
        'batch_size': config.batch_size,
        'val_batch_size': config.val_batch_size,
        'in_shape': (config.n_past, 3, config.image_width, config.image_height),
        'pre_seq_length': config.n_past,
        'aft_seq_length': config.n_future,
        'total_length': config.n_past + config.n_future,
        'save_inference': True,
        'batch_to_save': 20,
        'do_inference': False,
    }

    # update the remaining model config with the custom training config
    model_config = update_config(model_config, custom_training_config)
    
    # set multi-process settings
    # setup_multi_processes(model_config)

    # create the experiment object
    exp = BaseExperiment(model_args)
    
    if model_args.dist:
        n_gpus_total = exp._world_size
        config.val_batch_size = int(config.val_batch_size / n_gpus_total)
        config.data_threads = int((config.data_threads + n_gpus_total) / n_gpus_total)
    else:
        config.data_threads = 2
        
    # Load testing data
    test_data = MB(
        config,
        task='test',
        data_root=config.test_root,
        gs_root=config.test_root,
        audio_root=config.test_root,
    )

    rand_idx = np.sort(np.random.randint(0, len(test_data) - 1, model_args.batch_to_save)) if model_args.save_inference else None
    print("Saving batches at indices: {}".format(rand_idx))
    
    # Creating dataloader
    if model_args.dist:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
    else:
        test_sampler = None

    if model_args.save_inference:
        # print("\nBatches to save: {}\n".format(rand_idx))
        subset = Subset(test_data, list(rand_idx))
        test_loader = DataLoader(
            subset,
            num_workers=config.data_threads,
            batch_size=config.val_batch_size,
            sampler=test_sampler,
            shuffle=False,
            pin_memory=False,
        )
    else:
        # Define the dataloaders using the test loaders as the train and val loaders are not used
        test_loader = DataLoader(
            test_data,
            num_workers=config.data_threads,
            batch_size=config.val_batch_size,
            sampler=test_sampler,
            shuffle=False,
            pin_memory=False,
        )

    
    # # enumerate the test loader to visualize the data
    # for idx, (batch_x, batch_y, mean_, std_) in enumerate(test_loader):
    #     if idx == 0:
    #         print(">" * 35, " Visualizing data ", "<" * 35)
    #         print("Batch x shape: {}".format(batch_x.shape))
    #         if len(mean_.shape) > 1 and len(std_.shape) > 1:
    #                 mean_, std_ = mean_.cpu().numpy(), std_.cpu().numpy()
    #                 data_mean, data_std = np.transpose(mean_, (0, 3, 1, 2)), np.transpose(std_, (0, 3, 1, 2))
    #                 data_mean, data_std = np.expand_dims(data_mean, axis=0), np.expand_dims(data_std, axis=0)
    #         print("Mean shape: {}".format(data_mean.shape))
    #         batch_x_np = batch_y.cpu().numpy()
    #         batch_x_np = (batch_x_np * data_std ) + data_mean
    #         batch_x_np = batch_x_np * 255.0
    #         batch_x_np = batch_x_np.astype(np.uint8)
    #         # print(batch_x_np[0][0])
    #         show_video_line(batch_x_np[0], ncols=config.n_past, vmax=None, cbar=True, format='png', use_rgb=True, out_path='./custom/batch_x.png')

    #         break

    exp.init_experiment(dataloaders=(test_loader, test_loader, test_loader))

    print(">" * 35, " Testing ", "<" * 35)
    mse = exp.test()

    # print(">" * 35, " Testing finished ", "<" * 35)
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
