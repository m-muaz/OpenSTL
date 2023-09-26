# dataloaders for yt data
dataname = "custom_yt"
val_batch_size =  1
batch_size = 2
train_root = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/train"
val_root = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/val"
test_root = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/test"
# train_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/train"
# val_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=256
image_width=256

# model hyperparameters
n_past = 5 # changed 5 -> 2
n_future = 10 # changed 5 -> 2 
n_eval = n_past + n_future

# parameter to use previous audio features
ad_prev_frames = 0

# save the model statistics under this folder
model_save_path = "custom/data_stats/"