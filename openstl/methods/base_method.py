from typing import Dict, List, Union
import numpy as np
import os
import os.path as osp

import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar

# Adding tensorboard support for easy visualization
from tensorboardX import SummaryWriter

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank, self.world_size = get_dist_info()
            assert self.rank == int(device.split(':')[-1])
        else:
            self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        # setup metrics
        if 'weather' in self.args.dataname:
            self.metric_list, self.spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            self.metric_list, self.spatial_norm = ['mse', 'mae'], False

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
               print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

    def train_one_epoch(self, runner, train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def _dist_forward_collect(self, data_loader, metric_list=None, length=None, gather_data=False, **kwargs):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.
            **kwargs: Additional arguments.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))
        
        # Steps to sample validation dataloader
        valSteps = kwargs['valSteps'] if 'valSteps' in kwargs else 1
        valSteps = (valSteps - 1) if valSteps > 1 else 1

        # loop
        for idx, (batch_x, batch_y, batch_ad, mean, std) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            
            if idx % valSteps == 0:  # Use batch every valSteps (e.g. 10) epochs
                with torch.no_grad():
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_y = self._predict(batch_x, batch_ad, batch_y)
                
                data_mean, data_std = mean.cpu().numpy(), std.cpu().numpy()
                if len(data_mean.shape) > 1 and len(data_std.shape) > 1:
                    data_mean, data_std = np.transpose(data_mean, (0, 3, 1, 2)), np.transpose(data_std, (0, 3, 1, 2))
                    data_mean, data_std = np.expand_dims(data_mean, axis=0), np.expand_dims(data_std, axis=0)
                    # data_mean = np.squeeze(mean.cpu().numpy()) 
                    # data_std = np.squeeze(std.cpu().numpy())

                if gather_data:  # return raw datas
                    results.append(dict(zip(['inputs', 'preds', 'trues'],
                                            [batch_x.cpu().numpy(), batch_ad.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
                else:  # return metrics
                    eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                        data_mean, data_std,
                                        metrics=self.metric_list if metric_list is None else metric_list, 
                                        spatial_norm=self.spatial_norm, return_log=False)
                    eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                    for k in eval_res.keys():
                        if type(eval_res[k]) == list:
                            eval_res[k] = [val.reshape(1) for val in eval_res[k]]
                        else:
                            eval_res[k] = eval_res[k].reshape(1)
                    results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        results_all = {}
        for k in results[0].keys():
            if type(results[0][k]) == list:
                results_all[k] = []
                for i in range(len(results[0][k])):
                    results_cat = np.concatenate([batch[k][i] for batch in results], axis=0)
                    # gether tensors by GPU (it's no need to empty cache)
                    results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
                    results_strip = np.concatenate(results_gathered, axis=0)[:length]
                    results_all[k].append(results_strip)
            else:
                results_cat = np.concatenate([batch[k] for batch in results], axis=0)
                # gether tensors by GPU (it's no need to empty cache)
                results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
                results_strip = np.concatenate(results_gathered, axis=0)[:length]
                results_all[k] = results_strip

        return results_all

    def _nondist_forward_collect(self, data_loader, metric_list=None, length=None, gather_data=False, **kwargs):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.
            **kwargs: Additional arguments.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        resulting_images = []
        prog_bar = ProgressBar(len(data_loader))

        # Variables that control saving of inference results
        save_inference = kwargs['save_inference'] if 'save_inference' in kwargs else False
        # batch_to_save = kwargs['batch_to_save'] if save_inference else None
        # do_inference = kwargs['do_inference'] if 'do_inference' in kwargs else True

        # val steps to sample validation dataloader
        valSteps = kwargs['valSteps'] if 'valSteps' in kwargs else 1
        valSteps = (valSteps - 1) if valSteps > 1 else 1

        # zyhe: is this variable useful?
        length = len(data_loader.dataset) if length is None else length

        # New feature: Tensorboard support
        writer = kwargs['writer'] if 'writer' in kwargs else None

        # random idx to start saving images
        rand_idx = np.random.randint(0, len(data_loader.dataset) - 1)
        num_images = 100
        # make sure rand_idx is num_images far from the end of the dataset
        if rand_idx > len(data_loader.dataset) - num_images:
            rand_idx = rand_idx - num_images

        # loop
        for idx, (batch_x, batch_y, batch_ad, mean, std) in enumerate(data_loader):
            # print(f"Index {idx}")
            with torch.no_grad():
                if idx % valSteps == 0:  # Use batch every valSteps (e.g. 10) epochs
                    # print(f"Index {idx}")
                    # batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    batch_x, batch_y, batch_ad = batch_x.to(self.device), batch_y.to(self.device), batch_ad.to(self.device)
                    pred_y = self._predict(batch_x, batch_ad, batch_y).cpu()
                    # print(f"pred_y shape: {pred_y.shape}")
                
                    data_mean, data_std = mean.cpu().numpy(), std.cpu().numpy()
                    dm, ds  = data_mean.shape, data_std.shape
                    if len(data_mean.shape) > 1 and len(data_std.shape) > 1:
                        data_mean, data_std = data_mean.reshape(dm[0], 1, dm[1], dm[2], dm[3]), data_std.reshape(ds[0], 1, ds[1], ds[2], ds[3])
                        data_mean, data_std = np.transpose(data_mean, (0, 1, 4, 2, 3)), np.transpose(data_std, (0,1, 4, 2, 3))
                        # data_mean, data_std = np.expand_dims(data_mean, axis=1), np.expand_dims(data_std, axis=1)
                        # data_mean = np.squeeze(mean.cpu().numpy()) 
                        # data_std = np.squeeze(std.cpu().numpy())

                    if save_inference:
                        batch_x_save = batch_x.cpu().numpy() * data_std + data_mean
                        pred_y_save = pred_y.cpu().numpy() * data_std + data_mean
                        batch_y_save = batch_y.cpu().numpy() * data_std + data_mean
                        resulting_images.append(dict(zip(['inputs', 'preds', 'trues'],
                                            [batch_x_save, pred_y_save, batch_y_save])))
                    else:
                        if gather_data:  # return raw datas
                            # print("gather data at index {}".format(idx))
                            results.append(dict(zip(['inputs', 'preds', 'trues'],
                                                    [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
                        else:  # return metrics
                            eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                                data_mean, data_std,
                                                metrics=self.metric_list if metric_list is None else metric_list, 
                                                spatial_norm=self.spatial_norm, return_log=False)
                            eval_res['loss'] = self.criterion(pred_y.to(self.device), batch_y.to(self.device)).cpu().numpy()
                            for k in eval_res.keys():
                                if type(eval_res[k]) == list:
                                    eval_res[k] = [val.reshape(1) for val in eval_res[k]]
                                else:
                                    eval_res[k] = eval_res[k].reshape(1)
                                # Add resutls to log file for tensorboard
                                if writer is not None:
                                    # check if it is a list of scalars
                                    if type(eval_res[k]) == list:
                                        for i, val in enumerate(eval_res[k]):
                                            writer.add_scalar(f"{k}_{i}", val, idx)
                                    else:
                                        writer.add_scalar(k, eval_res[k], idx)
                            results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                # print("empty cache at index {}".format(idx))
                torch.cuda.empty_cache()
            # print("-"*50)

        # Saving the sampled images


        # post gather tensors"
        results_all = {}
        for k in results[0].keys():
            if type(results[0][k]) == list:
                results_all[k] = []
                for i in range(len(results[0][k])):
                    results_all[k].append(np.concatenate([batch[k][i] for batch in results], axis=0))
            else:
                results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        
        # resulting_images_all = {}
        # for key in resulting_images[0].keys():
        #     resulting_images_all[key] = np.concatenate([batch[key] for batch in resulting_images], axis=0)
            
        # return (results_all, resulting_images_all)
        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        # step to sample validation dataloader
        valSteps = kwargs['valStep'] if 'valStep' in kwargs else 10


        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(data_loader=vali_loader, length=len(vali_loader.dataset), gather_data=False,\
                                                 valSteps=valSteps)
        else:
            results = self._nondist_forward_collect(data_loader=vali_loader, length=len(vali_loader.dataset), gather_data=False,
                                                    save_inference=kwargs['save_inference'], valSteps=valSteps)
        results = results[0] if isinstance(results, tuple) else results

        eval_log = ""
        for k, v in results.items():
            v = v.mean()
            if k != "loss":
                eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
                eval_log += eval_str

        return results, eval_log

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        # Creating summary writer for tensorboard
        model_ckpt_path = kwargs['model_ckpt_path'] if 'model_ckpt_path' in kwargs else None
        use_tensorboard = kwargs['use_tensorboard'] if 'use_tensorboard' in kwargs else False
        if use_tensorboard and model_ckpt_path is not None:
            if self.dist:
                TB_name = osp.join(model_ckpt_path,"logDist")
            else:
                TB_name = osp.join(model_ckpt_path,"logNonDist")
        
        # do not create tensorboard for dist testing
        if self.dist:
            writer = None
        else:
            writer = SummaryWriter(TB_name)

        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(data_loader=test_loader, metric_list=kwargs['metric_list'], gather_data=False, writer=writer)
        else:
            results = self._nondist_forward_collect(data_loader=test_loader, metric_list=kwargs['metric_list'], gather_data=False, writer=writer)

        metric_results = results[0] if isinstance(results, tuple) else results
        # metric_results = results[0] if len(results) > 1 else results

        eval_log = ""
        for k, v in metric_results.items():
            if type(v) == list:
                if k != "loss":
                    eval_str = f"{k}:{[val.mean() for val in v]}" if len(eval_log) == 0 else f", {k}:{[val.mean() for val in v]}"
                    eval_log += eval_str
            else:
                v = v.mean()
                if k != "loss":
                    eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
                    eval_log += eval_str
        
        return results, eval_log

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."
