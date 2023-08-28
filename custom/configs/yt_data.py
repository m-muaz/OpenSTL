# dataloaders for yt data
dataname = "custom_yt"
val_batch_size =  1
batch_size = 2
train_root = "/home/v-xiaoxuanhe/zyhe/zhaoyuan/Interview/train"
val_root = "/home/v-xiaoxuanhe/zyhe/zhaoyuan/Interview/val"
test_root = "/home/v-xiaoxuanhe/zyhe/zhaoyuan/Interview/test"
# train_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/train"
# val_gs = "/mnt/sda1/muaz/ted_video/youtube-categories/ExtractedData/Interview/test"
data_threads = 8
image_height=544
image_width=960

# model hyperparameters
n_past = 2
n_future = 2
n_eval = 4

# parameter to use previous audio features
ad_prev_frames = 0

# save the model statistics under this folder
model_save_path = "custom/data_stats/"