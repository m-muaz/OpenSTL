import csv
import time
import numpy as np
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


# Class to log data into a csv file
class CsvLogger:
    def __init__(self, csvFilePath) -> None:
        self.file_name = csvFilePath
    
    def setHeader(self, header: list[str]) -> None:
        with open(self.file_name, 'a', newline='') as csvfile:
            # Create a csv writer
            csv_writer = csv.writer(csvfile)

            # Write the header 
            csv_writer.writerow(header)

    def csvEmpty(self) -> bool:
        try:
            with open(self.file_name, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                if len(list(csv_reader)) == 0:
                    return True
                else:
                    return False   
        except FileNotFoundError:
            return True

    
    def log(self, level, header:list, message: dict) -> None:
        with open(self.file_name, 'a', newline='') as csvfile:
            # Create a csv writer
            csv_writer = csv.writer(csvfile)

            loggedMsg = []
            # convert the data into a list message
            for col in header:
                if col in ['Level']:
                    loggedMsg.append(level)
                elif col in ['Timestamp']:
                    loggedMsg.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
                else:
                    # Extract relevant data from the dictionary of the results
                    data_val = message[col]
                    # Do appropriate processing depending if it is a list of array or an array
                    if isinstance(data_val, list) and all(isinstance(x, np.ndarray) for x in data_val):
                        data_val = [v.mean() for v in data_val]
                        loggedMsg.append(np.asarray(data_val))
                    else:
                        data_val = np.asarray(data_val)
                        data_val = data_val.mean()
                        loggedMsg.append(data_val)
            # Write the message
            csv_writer.writerow(loggedMsg)