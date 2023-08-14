import os
import natsort
import numpy as np
import cv2

import torch

# import torchaudio
from torch.utils import data

# from data.dataset_utils import StaticRandomCrop

FRAME_RATE = 30
AUDIO_SAMPLE_RATE = 16000


class MB(object):
    def __init__(
        self, args, data_root, gs_root, audio_root, train=False, transform=None
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

        assert os.path.exists(data_root)
        assert os.path.exists(gs_root)
        assert os.path.exists(audio_root)
        if self.train:
            self.start_index = 0

        # collect, colors, motion vectors, and depth
        self.ref = self.collectFileList(data_root)
        self.gs_ref = self.collectFileList(gs_root)
        # self.ad_ref = self.collectAudioFileList(audio_root)

        counts = [(len(el) - self.seq_len + 1) for el in self.ref]
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
                    if ad_sample_rate != AUDIO_SAMPLE_RATE:
                        ad_waveform = torchaudio.functional.resample(
                            ad_waveform, ad_sample_rate, AUDIO_SAMPLE_RATE
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
        gs_list = self.gs_ref[dataset_index - 1]
        # ad_data = self.ad_ref[dataset_index - 1][0]
        num_audio_frames = int(AUDIO_SAMPLE_RATE * (1 / FRAME_RATE))

        input_video_files = [
            image_list[index + offset] for offset in range(self.seq_len)
        ]
        input_gs_files = [gs_list[index + offset] for offset in range(self.seq_len)]
        # input_ads = [
        #     ad_data[
        #         :,
        #         (index + offset)
        #         * num_audio_frames : (index + offset + 1)
        #         * num_audio_frames,
        #     ]
        #     for offset in range(self.seq_len)
        # ]

        images = [
            cv2.resize(cv2.imread(imfile), (1920, 1080)) for imfile in input_video_files
        ]
        images_270p = [cv2.resize(imfile, (480, 270)) for imfile in images]
        # gs = [np.expand_dims(cv2.imread(imfile, 0), axis=2) for imfile in input_gs_files]
        gs = [cv2.resize(cv2.imread(imfile), (128, 64)) for imfile in input_video_files]
        input_shape = images[0].shape[:2]

        # if self.train:
        #     cropper = StaticRandomCrop(self.crop_size, input_shape)
        #     images = map(cropper, images)
        #     images = list(images)
        #     gs = map(cropper, gs)
        #     gs = list(gs)

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
        input_images_270p = np.stack(
            [im.astype(float) / 255.0 for im in images_270p], axis=0
        )
        input_gs = np.stack([im.astype(float) / 255.0 for im in gs], axis=0)
        # input_audios = np.stack([im.astype(float) for im in input_ads], axis=0)

        # return (pre_seq, post_seq) pairs from batch generator
        # return input_images, input_images_270p, input_gs, input_audios

        input_images_reshaped = input_images.transpose(0, 3, 1, 2)

        return (
            input_images_reshaped[: self.input_length, :, :, :],
            input_images_reshaped[self.input_length :, :, :, :]
        )
