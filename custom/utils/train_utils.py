from .main_utils import (normalize_data, sequence_input)
def get_batch(dtype, dataloader):
    while True:
        for image_sequence, image_small_sequence,  gs_sequence, audio_sequence in dataloader:
            print("image_sequence: ", image_sequence.shape)
            batch = normalize_data(dtype, image_sequence)
            batch_small = normalize_data(dtype, image_small_sequence)
            gs_batch = normalize_data(dtype, gs_sequence)
            ad_batch = sequence_input(audio_sequence.transpose_(0, 1), dtype)
            yield batch, batch_small, gs_batch, ad_batch
