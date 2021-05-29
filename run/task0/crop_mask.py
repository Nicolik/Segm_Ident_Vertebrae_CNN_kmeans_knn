import os
import argparse
import SimpleITK as sitk
from skimage import measure
import numpy as np
from config.paths import base_dataset_dir


def get_bbox(image_sitk):
    image_np = sitk.GetArrayFromImage(image_sitk)
    image_np[image_np != 0] = 1
    image_np = measure.label(image_np)
    print(np.unique(image_np))
    reg_prop = measure.regionprops(image_np)
    bb = reg_prop[0].bbox
    return bb


def cut_image(image_sitk, bbox):
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    image_np = sitk.GetArrayFromImage(image_sitk)
    image_np_new = image_np[xmin:xmax, ymin:ymax, zmin:zmax]
    image_sitk_new = sitk.GetImageFromArray(image_np_new)
    image_sitk_new.SetDirection(image_sitk.GetDirection())
    image_sitk_new.SetOrigin(image_sitk.GetOrigin())
    image_sitk_new.SetSpacing(image_sitk.GetSpacing())
    return image_sitk_new


def threshold_based_crop_and_bg_median(image, original_image):
    '''
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box and compute the background
    median intensity.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
        Background median intensity value.
    '''
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)

    # Get the median background intensity
    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image, image)
    bg_mean = label_intensity_stats_filter.GetMedian(inside_value)

    # Get the bounding box of the anatomy
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return bg_mean, sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box) / 2):],
                                          bounding_box[0:int(len(bounding_box) / 2)]), \
           sitk.RegionOfInterest(original_image, bounding_box[int(len(bounding_box) / 2):],
                                          bounding_box[0:int(len(bounding_box) / 2)]),


def crop_image_folder(images_folder, labels_folder, images_folder_save, labels_folder_save):
    images_folder = os.path.join(base_dataset_dir, images_folder)
    labels_folder = os.path.join(base_dataset_dir, labels_folder)
    images_folder_save = os.path.join(base_dataset_dir, images_folder_save)
    labels_folder_save = os.path.join(base_dataset_dir, labels_folder_save)

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder_save, exist_ok=True)
    os.makedirs(labels_folder_save, exist_ok=True)

    for msk in os.listdir(labels_folder):
        name = msk[:-19]
        image = os.path.join(images_folder, name + 'ct.nii.gz')
        mask = os.path.join(labels_folder, msk)
        print(mask)
        print(image)
        msk_load = sitk.ReadImage(mask)
        img_load = sitk.ReadImage(image)
        bbox = get_bbox(msk_load)
        print("BBox = {}".format(bbox))
        zmin, ymin, xmin, zmax, ymax, xmax = bbox
        print("z = [{} - {}] | y = [{} - {}] | x = [{} - {}]".format(zmin, zmax, ymin, ymax, xmin, xmax))
        modified_data = cut_image(msk_load, bbox)
        modified_data2 = cut_image(img_load, bbox)
        print("Before cut Image Size = {}".format(img_load.GetSize()))
        print("After  cut Image Size = {}".format(modified_data2.GetSize()))

        save_path_img = os.path.join(images_folder_save, name + 'ct.nii.gz')
        save_path_msk = os.path.join(labels_folder_save, msk)

        sitk.WriteImage(modified_data, save_path_msk)
        sitk.WriteImage(modified_data2, save_path_img)


def run(training_params, validation_params):
    crop_image_folder(**training_params)
    crop_image_folder(**validation_params)


path_to_original_images_ = "original_training"
path_to_original_labels_ = "original_training_mask"
path_to_cropped_images_  = "training_c"
path_to_cropped_labels_  = "training_mask_c"

path_to_validation_images_    = "original_validation"
path_to_validation_labels_    = "original_validation_mask"
path_to_cropped_valid_images_ = "validation_c"
path_to_cropped_valid_labels_ = "validation_mask_c"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for cropping images and labels around the spine mask")
    parser.add_argument(
        "-oti",
        "--original-training-images",
        default=path_to_original_images_, type=str,
        help="Specify the path where there are the original images"
    )
    parser.add_argument(
        "-otl",
        "--original-training-labels",
        default=path_to_original_images_, type=str,
        help="Specify the path where there are the original labels"
    )
    parser.add_argument(
        "-ovi",
        "--original-validation-images",
        default=path_to_validation_images_, type=str,
        help="Specify the path where to put the original validation images"
    )
    parser.add_argument(
        "-ovl",
        "--original-validation-labels",
        default=path_to_validation_labels_, type=str,
        help="Specify the path where to put the original validation labels"
    )
    args = parser.parse_args()
    training_params_ = {
        "images_folder": args.original_training_images,
        "labels_folder": args.original_training_labels,
        "images_folder_save": path_to_cropped_images_,
        "labels_folder_save": path_to_cropped_labels_
    }
    validation_params_ = {
        "images_folder": args.original_validation_images,
        "labels_folder": args.original_validation_labels,
        "images_folder_save": path_to_cropped_valid_images_,
        "labels_folder_save": path_to_cropped_valid_labels_
    }
    run(training_params_, validation_params_)
