import copy
import torch

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from models.vnet3d import VNet3D
from semseg.data_loader import QueueDataLoaderTraining


def print_config(config):
    attributes_config = [attr for attr in dir(config)
                         if not attr.startswith('__')]
    print("Config")
    for item in attributes_config:
        attr_val = getattr(config, item)
        if len(str(attr_val)) < 100:
            print("{:15s} ==> {}".format(item, attr_val))
        else:
            print("{:15s} ==> String too long [{} characters]".format(item, len(str(attr_val))))


def check_train_set(config):
    num_train_images = len(config.train_images)
    num_train_labels = len(config.train_labels)

    assert num_train_images == num_train_labels, "Mismatch in number of training images and labels!"

    print("There are: {} Training Images".format(num_train_images))
    print("There are: {} Training Labels".format(num_train_labels))


def check_torch_loader(config, check_net=False):
    train_data_loader_3D = QueueDataLoaderTraining(config)
    iterable_data_loader = iter(train_data_loader_3D)
    el = next(iterable_data_loader)
    inputs, labels = el['t1']['data'], el['label']['data']
    print("Shape of Batch: [input {}] [label {}]".format(inputs.shape, labels.shape))
    if check_net:
        net = VNet3D(num_outs=config.num_outs, channels=config.num_channels)
        outputs = net(inputs)
        print("Shape of Output: [output {}]".format(outputs.shape))


def print_folder(idx, train_index, val_index):
    print("+==================+")
    print("+ Cross Validation +")
    print("+     Folder {:d}     +".format(idx))
    print("+==================+")
    print("TRAIN [Images: {:3d}]:\n{}".format(len(train_index), train_index))
    print("VAL   [Images: {:3d}]:\n{}".format(len(val_index), val_index))


def print_test():
    print("+============+")
    print("+   Test     +")
    print("+============+")


def train_val_split(train_images, train_labels, train_index, val_index):
    train_images_np, train_labels_np = np.array(train_images), np.array(train_labels)
    train_images_list = list(train_images_np[train_index])
    val_images_list = list(train_images_np[val_index])
    train_labels_list = list(train_labels_np[train_index])
    val_labels_list = list(train_labels_np[val_index])
    return train_images_list, val_images_list, train_labels_list, val_labels_list


def train_val_split_config(config, train_index, val_index):
    train_images_list, val_images_list, train_labels_list, val_labels_list = \
        train_val_split(config.train_images, config.train_labels, train_index, val_index)
    new_config = copy.copy(config)
    new_config.train_images, new_config.val_images = train_images_list, val_images_list
    new_config.train_labels, new_config.val_labels = train_labels_list, val_labels_list
    return new_config


def nii_load(train_image_path):
    train_image_nii = nib.load(str(train_image_path), mmap=False)
    train_image_np = train_image_nii.get_fdata(dtype=np.float32)
    affine = train_image_nii.affine
    return train_image_np, affine


def sitk_load(train_image_path):
    train_image_sitk = sitk.ReadImage(train_image_path)
    train_image_np = sitk.GetArrayFromImage(train_image_sitk)
    origin, spacing, direction = train_image_sitk.GetOrigin(), \
                                 train_image_sitk.GetSpacing(), train_image_sitk.GetDirection()
    meta_sitk = {
        'origin': origin,
        'spacing': spacing,
        'direction': direction
    }
    return train_image_np, meta_sitk


def nii_write(outputs_np, affine, filename_out):
    outputs_nib = nib.Nifti1Image(outputs_np, affine)
    outputs_nib.header['qform_code'] = 1
    outputs_nib.header['sform_code'] = 0
    outputs_nib.to_filename(filename_out)


def sitk_write(outputs_np, meta_sitk, filename_out):
    outputs_sitk = sitk.GetImageFromArray(outputs_np)
    outputs_sitk.SetDirection(meta_sitk['direction'])
    outputs_sitk.SetSpacing(meta_sitk['spacing'])
    outputs_sitk.SetOrigin(meta_sitk['origin'])
    sitk.WriteImage(outputs_sitk, filename_out)


def torch5d_to_np3d(outputs, original_shape):
    outputs = torch.argmax(outputs, dim=1)  # 1 x Z x Y x X
    outputs_np = outputs.data.cpu().numpy()
    outputs_np = outputs_np[0]  # Z x Y x X
    outputs_np = outputs_np[:original_shape[0], :original_shape[1], :original_shape[2]]
    outputs_np = outputs_np.astype(np.uint8)
    return outputs_np


def plot_confusion_matrix(
        cm,
        target_names=None,
        title='Confusion matrix',
        cmap=None,
        normalize=True,
        already_normalized=False,
        path_out=None
):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.matshow(cm, cmap=cmap)
    plt.title(title, pad=25.)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize or already_normalized else cm.max() / 2
    print("Thresh = {}".format(thresh))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize or already_normalized:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if path_out is not None:
        plt.savefig(path_out)
    plt.show()
