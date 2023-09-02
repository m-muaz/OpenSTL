method = "SimVP"
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = "gSTA"
hid_S = 64
hid_T = 256
N_T = 3
N_S = 6
fp16=False
# training
lr = 1e-4
drop_path = 0
sched = "onecycle"
epoch = 1
log_step = 1
# model checkpoint
ex_name = f"custom_yt_data_normalized_{epoch}_epochs_audio_baseline_SimVP_gSTA"
