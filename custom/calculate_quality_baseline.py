import warnings

warnings.filterwarnings("ignore")

import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import torch
from torch.utils.data import DataLoader

from openstl.utils import create_parser, load_config, print_log, check_dir, ProgressBar
from openstl.core import metric
from mb_multifile_debug import MB
from custom.utils import load_dtparser, update_config


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)


def calculate_quality_baseline(test_loader):
    prog_bar = ProgressBar(len(test_loader))
    results = []
    for idx, (batch_x, batch_y, mean, std) in enumerate(test_loader):
        if idx % test_loader.dataset.seq_len == 0:
            data_mean, data_std = mean.numpy(), std.numpy()
            if len(data_mean.shape) > 1 and len(data_std.shape) > 1:
                data_mean, data_std = np.transpose(data_mean, (0, 3, 1, 2)), np.transpose(data_std, (0, 3, 1, 2))
                data_mean, data_std = np.expand_dims(data_mean, axis=0), np.expand_dims(data_std, axis=0)

            duplicated_arr_x = np.repeat(batch_x.numpy(), test_loader.dataset.output_length, axis=1)
            eval_res, _ = metric(duplicated_arr_x, batch_y.numpy(),
                                data_mean, data_std,
                                metrics=['ssim', 'psnr'], 
                                spatial_norm=False, return_log=False)
            for k in eval_res.keys():
                eval_res[k] = [np.array(val).reshape(1) for val in eval_res[k]]
            results.append(eval_res)
            prog_bar.update()
    
    metric_results = {}
    for k in results[0].keys():
        if type(results[0][k]) == list:
            metric_results[k] = []
            for i in range(len(results[0][k])):
                metric_results[k].append(np.concatenate([batch[k][i] for batch in results], axis=0))

    eval_log = ""
    for k, v in metric_results.items():
        if k != "loss":
            eval_str = f"{k}:{[val.mean() for val in v]}" if len(eval_log) == 0 else f", {k}:{[val.mean() for val in v]}"
            eval_log += eval_str
    
    return metric_results, eval_log


try:
    args = load_dtparser().parse_args()
    opt = args.__dict__

    # Create load_config object
    print(">" * 70)
    config = update_config(opt, load_config("./custom/configs/yt_data.py"))
    config = Config(config)
    config.dtype = torch.cuda.FloatTensor
    
    config.n_past = 1
    config.n_future = 9
    config.n_eval = 10

    # Load data
    test_data = MB(
        config,
        task='test',
        data_root=config.test_root,
        gs_root=config.test_root,
        audio_root=config.test_root,
    )

    # Creating dataloader for non-distributed training
    test_sampler = None
    config.data_threads = 2

    # Define the dataloaders
    test_loader = DataLoader(
        test_data,
        num_workers=config.data_threads,
        batch_size=config.val_batch_size,
        sampler=test_sampler,
        shuffle=False,
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

    metric_results, eval_log = calculate_quality_baseline(test_loader)
    
    path = 'custom/data_stats'
    print_log(eval_log)
    folder_path = osp.join(path, 'reuse_{}'.format(config.n_eval))
    check_dir(folder_path)
    
    for np_data in metric_results.keys():
        np.save(osp.join(folder_path, np_data + '_baseline.npy'), metric_results[np_data])
        
except Exception as e:
     print(">" * 35, " Testing failed with error", "<" * 35)
     print(e)
     