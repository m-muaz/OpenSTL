# dataloaders for yt data
val_batch_size =  1
batch_size = 1
train_root = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/train"
val_root = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=512
image_width=512

# model hyperparameters
n_past = 2
n_future = 2
n_eval = 4