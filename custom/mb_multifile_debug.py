import os
import natsort
import numpy as np
import cv2

import torch
import torchaudio
from torch.utils import data
from custom.utils import normalize_data, sequence_input
from torch.utils.data import Sampler


class CustomStartSampler(Sampler):
    def __init__(self, dataset, start_idx, shuffle=False):
        super(CustomStartSampler, self).__init__(dataset)
        self.dataset = dataset
        self.start_idx = start_idx
        self.shuffle = shuffle
        self.indices = list(range(self.start_idx, len(self.dataset)))

    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.indices)).tolist())
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
    
class MB(object):
    def __init__(
        self, args, data_root, train=False, transform=None
    ):
        self.train = train
        self.transform = transform
        self.chsize = 3
        # carry over command line arguments
        self.input_length = args.n_past
        self.output_length = args.n_future
        self.seq_len = self.input_length + self.output_length
        self.crop_size = [args.image_height, args.image_width]
        self.start_index = 0
        self.stride = args.stride
        self.dtype = args.dtype
        self.video_frame_rate = args.video_frame_rate
        self.audio_sample_rate = args.audio_sample_rate
        self.mean = None
        self.std = None

        assert os.path.exists(data_root)
        if self.train:
            self.start_index = 0

        # collect, colors, motion vectors, and depth
        self.ref = self.collectFileList(data_root)
        self.ad_ref = self.collectAudioFileList(data_root)
        self.ad_prev_len = 3

        counts = [(len(el) - self.seq_len) for el in self.ref]
        self.total = np.sum(counts)
        self.cum_sum = list(np.cumsum([0] + [el for el in counts]))

    def collectFileList(self, root):
        include_ext = [".png", ".jpg", "jpeg", ".bmp"]
        # collect subfolders, excluding hidden files, but following symlinks
        dirs = [
            x[0] for x in os.walk(root, followlinks=True) if not x[0].startswith(".")
        ]

        # naturally sort, both dirs and individual images, while skipping hidden files
        dirs = natsort.natsorted(dirs)

        datasets = [
            [
                os.path.join(fdir, el)
                for el in natsort.natsorted(os.listdir(fdir))
                if os.path.isfile(os.path.join(fdir, el))
                and not el.startswith(".")
                and any([el.endswith(ext) for ext in include_ext])
            ]
            for fdir in dirs
        ]

        return [el for el in datasets if el]

    def collectAudioFileList(self, root):
        include_ext = ["wav"]
        # collect subfolders, excluding hidden files, but following symlinks
        dirs = [
            x[0] for x in os.walk(root, followlinks=True) if not x[0].startswith(".")
        ]

        # naturally sort, both dirs and individual images, while skipping hidden files
        dirs = natsort.natsorted(dirs)

        datasets = []
        for fdir in dirs:
            each_data = []
            for el in natsort.natsorted(os.listdir(fdir)):
                if (
                    os.path.isfile(os.path.join(fdir, el))
                    and not el.startswith(".")
                    and any([el.endswith(ext) for ext in include_ext])
                ):
                    ad_waveform, ad_sample_rate = torchaudio.load(
                        os.path.join(fdir, el)
                    )
                    if ad_sample_rate != self.audio_sample_rate:
                        ad_waveform = torchaudio.functional.resample(
                            ad_waveform, ad_sample_rate, self.audio_sample_rate
                        ).numpy()
                    each_data.append(ad_waveform)
            datasets.append(each_data)

        return [el for el in datasets if el]

    def __len__(self):
        return int(self.total)

    def __getitem__(self, index):
        # adjust index
        index = len(self) + index if index < 0 else index
        index = index + self.start_index

        dataset_index = np.searchsorted(self.cum_sum, index + 1)
        index = index - self.cum_sum[np.maximum(0, dataset_index - 1)]

        image_list = self.ref[dataset_index - 1]
        ad_data = self.ad_ref[dataset_index - 1][0]
        num_audio_frames = int(self.audio_sample_rate * (1 / self.video_frame_rate))

        input_video_files = [
            image_list[index + offset] for offset in range(self.seq_len)
        ]
        # audio data has the previous frame sequence and the current audio frame
        input_ads = [
            ad_data[
                :,
                (index + offset - self.ad_prev_len)
                * num_audio_frames : (index + self.input_length + 1 + offset)
                * num_audio_frames,
            ]
            for offset in range(self.input_length)
        ]

        images = [
            cv2.resize(cv2.imread(imfile), (self.crop_size[1], self.crop_size[0])) for imfile in input_video_files
        ]
        input_shape = images[0].shape[:2]

        # Pad images along height and width to fit them evenly into models.
        input_shape = images[0].shape[:2]
        height, width = input_shape
        padded_height = height
        padded_width = width
        if (height % self.stride) != 0:
            padded_height = (height // self.stride + 1) * self.stride
            images = [
                np.pad(im, ((0, padded_height - height), (0, 0), (0, 0)), "reflect")
                for im in images
            ]
            gs = [
                np.pad(im, ((0, padded_height - height), (0, 0), (0, 0)), "reflect")
                for im in gs
            ]

        if (width % self.stride) != 0:
            padded_width = (width // self.stride + 1) * self.stride
            images = [
                np.pad(im, ((0, 0), (0, padded_width - width), (0, 0)), "reflect")
                for im in images
            ]
            gs = [
                np.pad(im, ((0, 0), (0, padded_width - width), (0, 0)), "reflect")
                for im in gs
            ]

        input_images = np.stack([im.astype(float) / 255.0 for im in images], axis=0)
        input_audios = np.stack([im.astype(float) for im in input_ads], axis=0)  # (seq_num, 2, num_audio_frames)

        # input_images_tensor = input_images.transpose(0, 3, 1, 2)
        input_images_tensor = torch.tensor(input_images, dtype=torch.float)
        input_images_tensor.transpose_(2,3)
        input_images_tensor.transpose_(1,2)
        
        input_audios_tensor = torch.tensor(input_audios, dtype=torch.float)

        # return input_images_tensor
        return (
            input_images_tensor[ :self.input_length, :, :, :],
            input_images_tensor[self.input_length: , :, :, :],
            input_audios_tensor
        )
