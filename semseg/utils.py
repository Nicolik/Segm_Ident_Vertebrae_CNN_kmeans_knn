import numpy as np
from medpy.metric.binary import hd, assd

def dice_coeff(gt, pred, eps=1e-5):
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def multi_dice_coeff(gt, pred, num_classes):
    labels = one_hot_encode_np(gt, num_classes)
    outputs = one_hot_encode_np(pred, num_classes)
    dices = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls]
        labels_  = labels[:, cls]
        dice_ = dice_coeff(outputs_, labels_)
        dices.append(dice_)
    return sum(dices) / (num_classes-1)


def multi_dice_coeff_with_list(gt, pred, num_classes, value_to_start):
    labels = one_hot_encode_np(gt, value_to_start + num_classes)
    outputs = one_hot_encode_np(pred, value_to_start + num_classes)
    dices = list()
    for cls in range(value_to_start, value_to_start + num_classes):
        outputs_ = outputs[:, cls]
        labels_  = labels[:, cls]
        dice_ = dice_coeff(outputs_, labels_)
        dices.append(dice_)
    return sum(dices) / (num_classes), dices


def multi_hausdorff_distance(gt, pred, num_classes, start_value=1):
    labels = one_hot_encode_np(gt, start_value + num_classes - 1)
    outputs = one_hot_encode_np(pred, start_value + num_classes - 1)
    hausdorff_distances = list()
    for cls in range(start_value, start_value + num_classes - 1):
        outputs_ = outputs[:, cls]
        labels_ = labels[:, cls]
        hausdorff_distance_ = hd(outputs_, labels_)
        hausdorff_distances.append(hausdorff_distance_)
    return sum(hausdorff_distances) / (num_classes-1)


def multi_assd_distance(gt, pred, num_classes, start_value=1):
    labels = one_hot_encode_np(gt, start_value + num_classes - 1)
    outputs = one_hot_encode_np(pred, start_value + num_classes - 1)
    assd_distances = list()
    for cls in range(start_value,  start_value + num_classes - 1):
        outputs_ = outputs[:, cls]
        labels_ = labels[:, cls]
        assd_distance_ = assd(outputs_, labels_)
        assd_distances.append(assd_distance_)
    return sum(assd_distances) / (num_classes-1)


def multi_hausdorff_distance_with_list(gt, pred, num_classes, value_to_start):
    labels = one_hot_encode_np(gt, value_to_start + num_classes)
    outputs = one_hot_encode_np(pred, value_to_start + num_classes)
    hausdorff_distances = list()
    for cls in range(value_to_start, value_to_start + num_classes):
        outputs_ = outputs[:, cls]
        labels_ = labels[:, cls]
        hausdorff_distance_ = hd(outputs_, labels_)
        hausdorff_distances.append(hausdorff_distance_)
    return sum(hausdorff_distances) / (num_classes), hausdorff_distances


def multi_assd_distance_with_list(gt, pred, num_classes, value_to_start):
    labels = one_hot_encode_np(gt, value_to_start + num_classes)
    outputs = one_hot_encode_np(pred, value_to_start + num_classes)
    assd_distances = list()
    for cls in range(value_to_start, value_to_start + num_classes):
        outputs_ = outputs[:, cls]
        labels_ = labels[:, cls]
        assd_distance_ = assd(outputs_, labels_)
        assd_distances.append(assd_distance_)
    return sum(assd_distances) / (num_classes), assd_distances


def one_hot_encode_np(label, num_classes):
    """ Numpy One Hot Encode
    :param label: Numpy Array of shape BxHxW or BxDxHxW
    :param num_classes: K classes
    :return: label_ohe, Numpy Array of shape BxKxHxW or BxKxDxHxW
    """
    assert len(label.shape) == 3 or len(label.shape) == 4, 'Invalid Label Shape {}'.format(label.shape)
    label_ohe = None
    if len(label.shape) == 3:
        label_ohe = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]))
    elif len(label.shape) == 4:
        label_ohe = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2], label.shape[3]))
    for batch_idx, batch_el_label in enumerate(label):
        for cls in range(num_classes):
            label_ohe[batch_idx, cls] = (batch_el_label == cls)
    return label_ohe


def min_max_normalization(input):
    return (input - input.min()) / (input.max() - input.min())


def z_score_normalization(input):
    input_mean = np.mean(input)
    input_std = np.std(input)
    return (input - input_mean)/input_std


def zero_pad_3d_image(image, pad_ref=(64,64,64), value_to_pad = 0):
    if value_to_pad == 0:
        image_padded = np.zeros(pad_ref)
    else:
        image_padded = value_to_pad * np.ones(pad_ref)
    image_padded[:image.shape[0],:image.shape[1],:image.shape[2]] = image
    return image_padded
