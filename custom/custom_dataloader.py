import warnings

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from openstl.utils import (
    load_config,
    show_video_line,
    create_parser,
    setup_multi_processes,
)
from openstl.api import BaseExperiment
from mb_multifile_debug import CustomStartSampler, MB
from custom.utils import load_dtparser, update_config, sequence_input, normalize_data
from openstl.utils import update_config as upd


class Config:
    def __init__(self, *args):
        self.__dict__.update(*args)


def createDataloader(dl_config, dist=False):
    # Load training and testing data
    train_data = MB(
        dl_config,
        task="train",
        data_root=dl_config.train_root,
        gs_root=dl_config.train_root,
        audio_root=dl_config.train_root,
    )
    val_data = MB(
        dl_config,
        task="val",
        data_root=dl_config.val_root,
        gs_root=dl_config.val_root,
        audio_root=dl_config.val_root,
    )
    test_data = MB(
        dl_config,
        task="test",
        data_root=dl_config.test_root,
        gs_root=dl_config.test_root,
        audio_root=dl_config.test_root,
    )

    # Creating dataloader
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_data, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # Define the dataloaders
    train_loader = DataLoader(
        train_data,
        num_workers=dl_config.data_threads,
        batch_size=dl_config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        num_workers=dl_config.data_threads,
        batch_size=dl_config.batch_size,
        sampler=val_sampler,
        shuffle=(val_sampler is None),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        num_workers=dl_config.data_threads,
        batch_size=dl_config.val_batch_size,
        sampler=test_sampler,
        shuffle=(test_sampler is None),
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
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
    model_config = update_config(
        model_config, load_config("./custom/configs/SimVP_gSTA.py")
    )

    # update the remaining model config with the custom training config
    custom_training_config = {
        "batch_size": config.batch_size,
        "val_batch_size": config.val_batch_size,
        "in_shape": (
            config.n_past,
            3,  # 3 is RGB channels
            config.image_height,
            config.image_width,
            config.ad_prev_frames,  # previous frames for audio (if any)
            config.audio_sample_rate,
            config.video_frame_rate,
        ),
        "pre_seq_length": config.n_past,
        "aft_seq_length": config.n_future,
        "total_length": config.n_past + config.n_future,
        "auto_resume": True,
    }

    model_config = update_config(model_config, custom_training_config)

    # set multi-process settings
    # setup_multi_processes(model_config)

    # create the experiment object
    exp = BaseExperiment(model_args)

    if model_args.dist:
        n_gpus_total = exp._world_size
        config.batch_size = int(config.batch_size / n_gpus_total)
        exp.args.batch_size = config.batch_size
        config.data_threads = int((config.data_threads + n_gpus_total) / n_gpus_total)
    else:
        config.data_threads = 2

    train_loader, val_loader, test_loader = createDataloader(config, model_args.dist)

    exp.init_experiment(dataloaders=(train_loader, val_loader, test_loader))

    # clear cuda cache
    torch.cuda.empty_cache()

    # # run the experiment
    # print(">" * 35, " Training ", "<" * 35)
    # exp.train()

    # print(">" * 35, " Testing ", "<" * 35)
    # exp.test()
