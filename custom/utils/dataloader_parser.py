import argparse


def load_dtparser():
    # Create a parser object to ensure replication of AudioVideo Recovery Github codebase
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument(
        "--val_batch_size", default=1, type=int, help="validation batch size"
    )
    parser.add_argument(
        "--train_root", metavar="TRAINING_FILE", help="root directory for training data"
    )
    parser.add_argument(
        "--val_root",
        metavar="VALIDATION_FILE",
        help="root directory for validation data",
    )
    parser.add_argument(
        "--train_gs",
        metavar="TRAINING_GAMASTATES_FILE",
        help="root directory for training gamestates",
    )
    parser.add_argument(
        "--val_gs",
        metavar="VALIDATION_GAMASTATES_FILE",
        help="root directory for validation gamestates",
    )
    # parser.add_argument('--train_audio_root', metavar="TRAINING_FILE", help='root directory for training data')
    # parser.add_argument('--val_audio_root', metavar="VALIDATION_FILE", help='root directory for validation data')

    parser.add_argument(
        "--image_height",
        type=int,
        default=1080,
        metavar="CROP_SIZE",
        help="Spatial dimension to crop training samples for training (default : [448, 448])",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=1920,
        metavar="CROP_SIZE",
        help="Spatial dimension to crop training samples for training (default : [448, 448])",
    )
    parser.add_argument(
        "--val_height",
        type=int,
        default=1080,
        metavar="IMG_SIZE",
        help="Image height when inference",
    )
    parser.add_argument(
        "--val_width",
        type=int,
        default=1920,
        metavar="IMG_SIZE",
        help="Image width when inference",
    )
    parser.add_argument(
        "--stride",
        default=8,
        type=int,
        help="The factor for which padded validation image sizes should be evenly divisible. (default: 16)",
    )
    parser.add_argument(
        "--n_past", type=int, default=5, help="number of frames to condition on"
    )
    parser.add_argument(
        "--n_future", type=int, default=1, help="number of frames to predict"
    )
    parser.add_argument(
        "--n_eval", type=int, default=6, help="number of frames to predict at eval time"
    )
    parser.add_argument(
        "--data_threads", type=int, default=8, help="number of data loading threads"
    )
    return parser
