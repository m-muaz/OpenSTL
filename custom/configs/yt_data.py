# dataloaders for yt data
dataname = "custom_yt"
val_batch_size =  1
batch_size = 1
train_root = "/home/aiscuser/AudioVideo/data/Interview/train"
val_root = "/home/aiscuser/AudioVideo/data/Interview/val"
test_root = "/home/aiscuser/AudioVideo/data/Interview/test"
# train_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/train"
# val_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=544
image_width=960

# model hyperparameters
n_past = 10
n_future = 10
n_eval = 20

# parameter to use previous audio features
ad_prev_frames = 0

# save the model statistics under this folder
model_save_path = "custom/data_stats/"