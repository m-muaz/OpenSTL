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
        data_mean, data_std = mean.cpu().numpy(), std.cpu().numpy()
        data_mean, data_std = np.transpose(data_mean, (0, 3, 1, 2)), np.transpose(data_std, (0, 3, 1, 2))
        data_mean, data_std = np.expand_dims(data_mean, axis=0), np.expand_dims(data_std, axis=0)
        
        eval_res, _ = metric(batch_x.numpy(), batch_y.numpy(),
                            data_mean, data_std,
                            metrics=['ssim', 'psnr'], 
                            spatial_norm=False, return_log=False)
        for k in eval_res.keys():
            eval_res[k] = eval_res[k].reshape(1)
        results.append(eval_res)
        prog_bar.update()
        
    metric_results = {}
    for k in results[0].keys():
        metric_results[k] = np.concatenate([batch[k] for batch in results], axis=0)

    eval_log = ""
    for k, v in metric_results.items():
        v = v.mean()
        if k != "loss":
            eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
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
    config.n_future = 1
    config.n_eval = 2

    # Load data
    test_data = MB(
        config,
        train=False,
        data_root=config.val_root,
        gs_root=config.val_root,
        audio_root=config.val_root,
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

    metric_results, eval_log = calculate_quality_baseline(test_loader)
    
    base_dir = model_args.res_dir if model_args.res_dir is not None else 'work_dirs'
    path = osp.join(base_dir, model_args.ex_name if not model_args.ex_name.startswith(model_args.res_dir) \
            else model_args.ex_name.split(model_args.res_dir+'/')[-1])
    
    print_log(eval_log)
    folder_path = osp.join(path, 'saved')
    check_dir(folder_path)
    
    for np_data in metric_results.keys():
        np.save(osp.join(folder_path, np_data + '_baseline.npy'), metric_results[np_data])
        
    
except Exception as e:
     print(">" * 35, " Testing failed with error", "<" * 35)
     print(e)
     