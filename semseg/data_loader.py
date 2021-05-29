import torch
import torchio
from torchio import SubjectsDataset
from config.augm import train_transform


class SemSegConfig(object):
    train_images = None
    train_labels = None
    val_images   = None
    val_labels   = None
    do_normalize = True
    batch_size   = 8
    num_workers  = 8


def create_subject_list(images, labels):
    subject_list = []
    for idx, (image_path, label_path) in enumerate(zip(images, labels)):
        s1 = torchio.Subject(
            t1=torchio.Image(type=torchio.INTENSITY, path=image_path),
            label=torchio.Image(type=torchio.LABEL, path=label_path),
        )
        subject_list.append(s1)
    return subject_list


def QueueDataLoaderTraining(config):
    print('Building TorchIO Training Set Loader...')
    subject_list = create_subject_list(config.train_images, config.train_labels)
    subjects_dataset = SubjectsDataset(subject_list, transform=train_transform)
    patches_training_set = torchio.Queue(
        subjects_dataset=subjects_dataset,
        max_length=300,
        samples_per_volume=10,
        sampler=torchio.sampler.UniformSampler(64),
        num_workers=config.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    patch_loader = torch.utils.data.DataLoader(patches_training_set, batch_size=config.batch_size)
    print('TorchIO Training Loader built!')
    return patch_loader


def QueueDataLoaderValidation(config):
    print('Building TorchIO Validation Set Loader...')
    subject_list = create_subject_list(config.val_images, config.val_images)
    patches_validation_set = torchio.Queue(
        subjects_dataset=subject_list,
        max_length=300,
        samples_per_volume=10,
        sampler=torchio.sampler.UniformSampler(64),
        num_workers=config.num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )
    patch_loader = torch.utils.data.DataLoader(patches_validation_set, batch_size=config.batch_size)
    print('TorchIO Validation Loader built!')
    return patch_loader


def z_score_normalization(inputs):
    input_mean = torch.mean(inputs)
    input_std = torch.std(inputs)
    return (inputs - input_mean)/input_std
