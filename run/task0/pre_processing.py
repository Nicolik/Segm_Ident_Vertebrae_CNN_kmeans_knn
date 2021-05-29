import itk
import os
import numpy as np
from config.paths import base_dataset_dir


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input sitk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented


def smooth(image, sigma):
    """
    Smooth image with Gaussian smoothing.
    :param image: sitk image.
    :param sigma: Sigma for smoothing.
    :return: Smoothed image.
    """
    ImageType = itk.Image[itk.F, 3]
    filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetSigma(sigma)
    filter.Update()
    smoothed = filter.GetOutput()
    return smoothed


def clamp(image):
    """
    Clamp image between -1024 to 8192.
    :param image: sitk image.
    :return: Clamped image.
    """
    ImageType = itk.Image[itk.F, 3]
    filter = itk.ClampImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetBounds(-150, 1000)
    filter.Update()
    clamped = filter.GetOutput()
    return clamped


def pad_image(image):
    origin = image.GetOrigin()
    direction = image.GetDirection()
    spacing = image.GetSpacing()
    image_np = itk.GetArrayFromImage(image)
    shape = image_np.shape
    new_shape = list()
    to_pad = False
    for dim in shape:
        if dim < 64:
            to_pad = True
            new_shape.append(64)
        else:
            new_shape.append(dim)
    if to_pad:
        new_image = np.zeros(new_shape)
        new_image[:shape[0], :shape[1], :shape[2]] = image_np
        image = itk.GetImageFromArray(new_image)
        image.SetOrigin(origin)
        image.SetDirection(direction)
        image.SetSpacing(spacing)
    return image


def process_image(filename, input_folder, output_folder, sigma):
    """
    Reorient image at filename, smooth with sigma, clamp and save to output_folder.
    :param filename: The image filename.
    :param output_folder: The output folder.
    :param sigma: Sigma for smoothing.
    """
    basename = os.path.join(input_folder, filename)
    basename_wo_ext = basename[:basename.find('.nii.gz')]
    ImageType = itk.Image[itk.F, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(basename)
    image = reader.GetOutput()
    reoriented = reorient_to_rai(image)
    if not basename_wo_ext.endswith('_msk'):
        reoriented = smooth(reoriented, sigma)
        reoriented = clamp(reoriented)
    reoriented.SetOrigin([0, 0, 0])
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    reoriented.SetDirection(m)
    reoriented.SetSpacing([1,1,1])
    reoriented.Update()
    if not basename_wo_ext.endswith('_msk'):
        ImageType = itk.Image[itk.F, 3]
        normalizer = itk.RescaleIntensityImageFilter[ImageType, ImageType].New()
        normalizer.SetOutputMaximum(1)
        normalizer.SetOutputMinimum(0)
        normalizer.SetInput(reoriented)
        #normalizer.Update()
        reoriented = normalizer.GetOutput()
    reoriented = pad_image(reoriented)
    os.makedirs(output_folder, exist_ok=True)
    itk.imwrite(reoriented, os.path.join(output_folder, filename))


def run(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, output_mask_folder_bin):
    filename_images = os.listdir(input_image_folder)
    filename_masks = os.listdir(input_mask_folder)
    for filename_image, filename_mask in zip(filename_images, filename_masks):
        print('Image ' + filename_image + ' in processing...')
        process_image(filename_image, input_image_folder, output_image_folder, 0.75)
        print('Image ' + filename_image + ' processed')
        print('Mask  ' + filename_mask + ' in processing...')
        process_image(filename_mask, input_mask_folder, output_mask_folder, 0.75)
        print('Mask  ' + filename_mask + ' processed')
    labels_colored = os.listdir(output_mask_folder)
    os.makedirs(output_mask_folder_bin, exist_ok=True)
    for label_colored in labels_colored:
        path_to_label = os.path.join(output_mask_folder, label_colored)
        ImageType = itk.Image[itk.F, 3]
        reader = itk.ImageFileReader[ImageType].New()
        reader.SetFileName(path_to_label)
        image = reader.GetOutput()
        image_np = itk.GetArrayFromImage(image)
        image_np[image_np != 0] = 1
        image = itk.GetImageFromArray(image_np)
        image = pad_image(image)
        out_path = os.path.join(output_mask_folder_bin, label_colored)
        itk.imwrite(image, out_path)
        print('Binary mask saved to {}'.format(out_path))


if __name__ == "__main__":
    train_im = "training_c"
    train_msk = "training_mask_c"

    val_im = "validation_c"
    val_msk = "validation_mask_c"

    input_image_folder = os.path.join(base_dataset_dir, train_im)
    input_mask_folder = os.path.join(base_dataset_dir, train_msk)
    output_image_folder = os.path.join(base_dataset_dir, "training")
    output_mask_folder = os.path.join(base_dataset_dir, "training_mask_nb")
    output_mask_folder_bin = os.path.join(base_dataset_dir, "training_mask")

    os.makedirs(input_image_folder, exist_ok=True)
    os.makedirs(input_mask_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_mask_folder_bin, exist_ok=True)

    print("Training dataset preprocessing")
    run(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, output_mask_folder_bin)

    input_image_folder = os.path.join(base_dataset_dir, val_im)
    input_mask_folder = os.path.join(base_dataset_dir, val_msk)
    output_image_folder = os.path.join(base_dataset_dir, "validation")
    output_mask_folder = os.path.join(base_dataset_dir, "validation_mask_nb")
    output_mask_folder_bin = os.path.join(base_dataset_dir, "validation_mask")

    os.makedirs(input_image_folder, exist_ok=True)
    os.makedirs(input_mask_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_mask_folder_bin, exist_ok=True)

    print("Validation dataset preprocessing")
    run(input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, output_mask_folder_bin)
