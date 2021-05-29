import os
from config.paths import train_images_folder, train_labels_folder, train_images, train_labels
from semseg.data_loader import SemSegConfig


class SemSegMRIConfig(SemSegConfig):
    train_images = sorted([os.path.join(train_images_folder, train_image)
                           for train_image in train_images])
    train_labels = sorted([os.path.join(train_labels_folder, train_label)
                           for train_label in train_labels])
    val_images = None
    val_labels = None
    do_normalize = False
    batch_size = 8
    num_workers = 8
    lr = 0.0001
    # lr = 0.005
    epochs = 1000
    # epochs = 500
    low_lr_epoch = epochs // 5
    val_epochs = epochs // 5
    cuda = True
    num_outs = 2
    do_crossval = False
    num_folders = 2
    num_channels = 8
