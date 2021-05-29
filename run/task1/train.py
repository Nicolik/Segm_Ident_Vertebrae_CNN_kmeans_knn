import argparse
import os
import torch
import torch.optim as optim

from config.config_semseg import SemSegMRIConfig
from models.vnet3d import VNet3D
from config.paths import logs_folder
from run.utils import check_train_set, print_config, check_torch_loader, train_val_split_config
from semseg.data_loader import QueueDataLoaderTraining, QueueDataLoaderValidation
from semseg.train import train_model, val_model


def get_net(config):
    return VNet3D(num_outs=config.num_outs, channels=config.num_channels)


def run(config):
    ##########################
    # Check training set
    ##########################
    check_train_set(config)

    ##########################
    # Config
    ##########################
    print_config(config)
    index = round(len(config.train_images) * 9/10)
    print("Train set size = {}".format(index))
    print("Val   set size = {}".format(len(config.train_images)-index))
    train_index = list(range(index))
    val_index = list(range(index, len(config.train_images)))

    ##########################
    # Check Torch DataLoader and Net
    ##########################
    check_torch_loader(config, check_net=False)

    cuda_dev = torch.device('cuda')
    config_val = train_val_split_config(config, train_index, val_index)

    ##########################
    # Training 90%
    ##########################z
    net = get_net(config_val)
    config_val.lr = 0.0001
    # config_val.lr = 0.005
    optimizer = optim.Adam(net.parameters(), lr=config_val.lr)
    train_data_loader_3D = QueueDataLoaderTraining(config_val)
    net = train_model(net, optimizer, train_data_loader_3D,
                      config_val, device=cuda_dev, logs_folder=logs_folder)

    torch.save(net, os.path.join(logs_folder, "model.pt"))

    ##########################
    # Validation 10%
    ##########################
    val_data_loader_3D = QueueDataLoaderValidation(config_val)
    val_model(net, val_data_loader_3D, config_val, device=cuda_dev)


# python run/train.py
if __name__ == "__main__":
    config = SemSegMRIConfig()

    parser = argparse.ArgumentParser(description="Run Training for Vertebral Column Segmentation")
    parser.add_argument(
        "-e",
        "--epochs",
        default=config.epochs, type=int,
        help="Specify the number of epochs required for training"
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=config.batch_size, type=int,
        help="Specify the batch size"
    )
    parser.add_argument(
        "-v",
        "--val_epochs",
        default=config.val_epochs, type=int,
        help="Specify the number of validation epochs during training ** FOR FUTURE RELEASES **"
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=config.num_workers, type=int,
        help="Specify the number of workers"
    )
    parser.add_argument(
        "--lr",
        default=config.lr, type=float,
        help="Learning Rate"
    )

    args = parser.parse_args()
    config.epochs = args.epochs
    config.batch_size = args.batch
    config.val_epochs = args.val_epochs
    config.num_workers = args.workers
    config.lr = args.lr

    run(config)
