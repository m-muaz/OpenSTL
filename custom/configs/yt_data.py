# dataloaders for yt data
val_batch_size =  1
batch_size = 8
train_root = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/train"
val_root = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/test"
train_gs = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/train"
val_gs = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=240
image_width=428

# model hyperparameters
n_past = 3
n_future = 3
n_eval = 3