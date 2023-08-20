# dataloaders for yt data
dataname = "custom_yt"
val_batch_size =  1
batch_size = 2
train_root = "/data/data1/muaz/remote_fs/ted_video/youtube-categories/ExtractedData/Interview/train"
val_root = "/data/data1/muaz/remote_fs/ted_video/youtube-categories/ExtractedData/Interview/test"
train_gs = "/data/data1/muaz/remote_fs/ted_video/youtube-categories/ExtractedData/Interview/train"
val_gs = "/data/data1/muaz/remote_fs/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=512
image_width=512

# model hyperparameters
n_past = 2
n_future = 1
n_eval = 3

# save the model statistics under this folder
model_save_path = "/home/muaz/Documents/Audio_Video_Recovery/OpenSTL/custom/data_stats/"