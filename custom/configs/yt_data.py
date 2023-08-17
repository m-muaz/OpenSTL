# dataloaders for yt data
dataname = "custom_yt"
val_batch_size =  1
batch_size = 2
train_root = "/data/data1/muaz/ExtractedData/Interview/train"
val_root = "/data/data1/muaz/ExtractedData/Interview/test"
train_gs = "/data/data1/muaz/ExtractedData/Interview/train"
val_gs = "/data/data1/muaz/ExtractedData/Interview/test"
data_threads = 8
image_height=512
image_width=512

# model hyperparameters
n_past = 2
n_future = 1
n_eval = 3