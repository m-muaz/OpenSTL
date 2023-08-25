import os
import natsort
import numpy as np
import cv2
import time
import concurrent.futures
from tqdm import tqdm

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
        self, args, data_root, gs_root, audio_root, task, transform=None
    ):
        self.task = task
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
        
        # Dictionary to store means and std devs of datasets
        self.mean_dict = {}
        self.std_dict = {}

        assert os.path.exists(data_root)
        assert os.path.exists(gs_root)
        assert os.path.exists(audio_root)
        if self.task == 'train':
            self.start_index = 0

        # collect, colors, motion vectors, and depth
        self.ref = self.collectFileList(data_root)
        self.ad_ref = self.collectAudioFileList(data_root)
        self.ad_prev_len = 3

        self.counts = [(len(el) - self.seq_len) for el in self.ref]
        self.total = np.sum(self.counts)
        self.cum_sum = list(np.cumsum([0] + [el for el in self.counts]))

        # make dictionary to store length and dirname
        self._len_dirname = {(len(el) - self.seq_len + 1): os.path.basename(os.path.dirname(el[0])) for el in self.ref}

        # print len dir dict
        print(self._len_dirname)

        # Load mean and std of the dataset
        self.load_mean_std(args.model_save_path)
        # Check if mean_dict and std_dict are empty
        if not self.mean_dict and not self.std_dict:
            # Calculate mean and std
            self._data_mean()
            self._data_std_dev()
            # Save mean and std of the dataset
            self.save_mean_std(args.model_save_path)

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

    def _compute_mean_for_dir(self, dir_list):
        _placeholder_mean = 0.0
        num_images = len(dir_list)
        dirname = None
        with tqdm(total=num_images, desc="\033[92mComputing mean\033[0m") as pbar:
            for idx, img_path in enumerate(dir_list):
                if not dirname:
                    dirname = os.path.basename(os.path.dirname(img_path))
                img = cv2.imread(img_path)
                img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]))
                img = img.astype(float) / 255.0
                _placeholder_mean += img

                pbar.update(1)

        _placeholder_mean /= (len(dir_list))
        return (dirname, _placeholder_mean)

    # Function to compute standard deviation of each dir
    def _compute_std_for_dir(self, dir_list):
        _placeholder_std = 0.0
        num_images = len(dir_list)
        dirname = None
        with tqdm(total=num_images, desc="\033[92mComputing std dev\033[0m") as pbar:
            for idx, img_path in enumerate(dir_list):
                if not dirname:
                    dirname = os.path.basename(os.path.dirname(img_path))
                img = cv2.imread(img_path)
                img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]))
                img = img.astype(float) / 255.0
                mean = self.mean_dict[dirname]
                _placeholder_std += np.square(img - mean)

                pbar.update(1)

        _placeholder_std /= (len(dir_list))
        _placeholder_std = np.sqrt(_placeholder_std)
        return (dirname, _placeholder_std)

    # Compute mean of the dataset
    def _data_mean(self):
        print(">" * 35 + "Computing mean of the dataset" + ">" * 35)
        # dict to store mean of the dataset
        mean_list = {}

        # measure time taken to compute mean
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._compute_mean_for_dir, self.ref)
            # print(results.shape)

        for inner_result in results:
            print(inner_result)
            mean_list[inner_result[0]] = inner_result[1]
            # mean_list.append(inner_result)

        end_time = time.time()
        print(f"Time taken to compute mean: {end_time - start_time} seconds")

        self.mean_dict = mean_list
        # print(self.mean[0].shape)
        print([self.mean_dict[key].shape for key in self.mean_dict.keys()])
        print(">" * 35 + "Mean computation complete" + ">" * 35)

    # Compute the standard deviation of the dataset
    def _data_std_dev(self):
        print(">" * 35 + "Computing standard deviation of the dataset" + ">" * 35)
        # list to store standard deviation of the dataset
        std_list = {}

        # measure time taken to compute standard deviation
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(self._compute_std_for_dir, self.ref)

        for inner_result in results:
            std_list[inner_result[0]] = inner_result[1]
            # std_list.append(inner_result)

        end_time = time.time()
        print(
            f"Time taken to compute standard deviation: {end_time - start_time} seconds"
        )

        self.std_dict = std_list
        print(self.std_dict[key].shape for key in self.std_dict.keys())
        print(">" * 35 + "Standard deviation computation complete" + ">" * 35)

    # Method to save the mean and standard deviation of the dataset
    def save_mean_std(self, path):
        print(">" * 35 + "Saving dataset statistics" + ">" * 35)
        # Loop over the dir to get one image url
        dataset_statistics = {}
        for idx, dir in enumerate(self.ref):
            img_path = dir[0]
            # extract base name of the image
            basename = os.path.basename(os.path.dirname(img_path))

            # make basename the key of the dictionary and the value is a tuple of mean and std
            dataset_statistics[basename] = (self.mean_dict[basename], self.std_dict[basename])

        # save the dictionary as a numpy file if the file does not exist under the path
        if os.path.exists(path):
            # save the dictionary as a numpy file with name dataset_statistics.npy
            filename = os.path.join(path, f"dataset_statistics_{self.task}.npy")
            print(f"Saving dataset statistics to: {filename}")
            np.save(filename, dataset_statistics)
            print(dataset_statistics)
            print(">" * 35 + "Dataset statistics saved" + ">" * 35)

    # Method to load the mean and standard deviation of the dataset
    def load_mean_std(self, path):
        os.makedirs(path, exist_ok=True)
        # Check if the file exists under the path
        filename = os.path.join(path, f"dataset_statistics_{self.task}.npy")
        if os.path.exists(filename):
            # # make self.mean and self.std a list
            # self.mean = {}
            # self.std = {}
            # # load the dictionary as a numpy file
            dataset_statistics = np.load(filename, allow_pickle=True).item()
            # print(dataset_statistics)
            # loop over the dictionary to extract the mean and std
            for dir in self.ref:
                img_path = dir[0]
                basename = os.path.basename(os.path.dirname(img_path))
                self.mean_dict[basename] = dataset_statistics[basename][0]
                self.std_dict[basename] = dataset_statistics[basename][1]
                # self.mean.append(dataset_statistics[basename][0])
                # self.std.append(dataset_statistics[basename][1])
            print(">" * 35 + "Dataset statistics loaded" + ">" * 35)
        else:
            print(">" * 35 + "Dataset statistics not found" + ">" * 35)

    def __len__(self):
        return int(self.total)

    def __getitem__(self, index):
        # adjust index
        index = len(self) + index if index < 0 else index
        index = index + self.start_index

        dataset_index = np.searchsorted(self.cum_sum, index + 1)
        index = index - self.cum_sum[np.maximum(0, dataset_index - 1)]

        # Get the mean and std dev of the dataset
        dataset_len = self.counts[dataset_index - 1]
        # print("Dataset Index: ", dataset_index)
        # print("Dataset Len: ", dataset_len)
        
        dirname = self._len_dirname[dataset_len]
        mean = self.mean_dict[dirname] if dirname in self.mean_dict else 0.0
        std_dev = self.std_dict[dirname] if dirname in self.std_dict else 1.0
        # print("Mean shape , Std Dev shape: ", self.mean.shape, self.std_dev.shape)

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
            cv2.resize(cv2.imread(imfile), (self.crop_size[1], self.crop_size[0]))
            for imfile in input_video_files
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

        input_images = np.stack([(((im.astype(float) / 255.0) - mean) / (std_dev + np.finfo(float).eps)) for im in images], axis=0)
        input_audios = np.stack([im.astype(float) for im in input_ads], axis=0)  # (seq_num, 2, num_audio_frames)
        # input_images = np.stack([(im.astype(float) / 255.0) for im in images], axis=0)


        # input_audios = np.stack([im.astype(float) for im in input_ads], axis=0)

        # return (pre_seq, post_seq) pairs from batch generator
        # return input_images, input_gs, input_audios

        # input_images_tensor = input_images.transpose(0, 3, 1, 2)
        input_images_tensor = torch.tensor(input_images, dtype=torch.float)
        input_images_tensor.transpose_(2, 3)
        input_images_tensor.transpose_(1, 2)
        # self.mean = torch.mean(input_images_tensor, dim=1)
        # self.std = torch.std(input_images_tensor, dim=1)
        # input_images_reshaped = normalize_data(self.dtype, input_images_tensor)
        # input_images_tensor = torch.tensor(input_images_reshaped).float()
        # input_images_reshaped = sequence_input(input_images_tensor, self.dtype)
        
        input_audios_tensor = torch.tensor(input_audios, dtype=torch.float)

        # return input_images_tensor
        return (
            input_images_tensor[: self.input_length, :, :, :],
            input_images_tensor[self.input_length :, :, :, :],
            input_audios_tensor,
            mean,
            std_dev
        )
