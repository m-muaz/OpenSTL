# dataloaders for yt data
val_batch_size =  2
batch_size = 16
train_root = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/train"
val_root = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/test"
train_gs = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/train"
val_gs = "/mnt/sda1/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8

# model hyperparameters
n_past = 3
n_future = 3
n_eval = 3