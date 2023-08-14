from torch.autograd import Variable

# Function to update the config file
def update_config(args, config, exclude_keys=list()):
    """update the args dict with a new config"""
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if args.get(k, False):
            if args[k] != config[k] and k not in exclude_keys and args[k] is not None:
                print(f"overwrite config key -- {k}: {args[k]} -> {config[k]}")
                args[k] = config[k]
            else:
                continue
                # args[k] = config[k]
        else:
            args[k] = config[k]
    return args

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(dtype, sequence):
    # sequence.transpose_(0, 1)
    # sequence.transpose_(3, 4).transpose_(2, 3)

    # Updated code
    sequence.transpose_(2,3)
    sequence.transpose_(1,2)
    return sequence_input(sequence, dtype)
