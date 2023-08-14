method = "SimVP"
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = "gSTA"
hid_S = 64
hid_T = 256
N_T = 8
N_S = 4
# training
lr = 1e-3
drop_path = 0
sched = "onecycle"
epoch = 100
# model checkpoint
ex_name = "custom_yt_data_100_epochs_SimVP_gSTA"
