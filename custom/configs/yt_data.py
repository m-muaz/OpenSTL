# dataloaders for yt data
dataname = "custom_yt"
val_batch_size =  3*8
batch_size = 5*8
train_root = "/home/v-xiaoxuanhe/MultiModal/data/Interview/train"
val_root = "/home/v-xiaoxuanhe/MultiModal/data/Interview/val"
test_root = "/home/v-xiaoxuanhe/MultiModal/data/Interview/test"
# train_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/train"
# val_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=256
image_width=256

# model hyperparameters
n_past = 5 # changed 5 -> 2
n_future = 5 # changed 5 -> 2 
n_eval = n_future

# parameter to use previous audio features
ad_prev_frames = 0
ad_future_frames = 5

# save the model statistics under this folder
model_save_path = "custom/data_stats/"
