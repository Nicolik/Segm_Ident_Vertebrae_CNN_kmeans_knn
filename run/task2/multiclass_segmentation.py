import argparse
import os
import SimpleITK as sitk
import json
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, regionprops_table
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.cluster import KMeans
from semseg.utils import z_score_normalization, multi_dice_coeff_with_list, multi_hausdorff_distance_with_list, multi_assd_distance_with_list
from config.map import get_cmap, get_input_map
from config.paths import train_labels, train_labels_folder, multiclass_prediction_folder


def run(input_path, gt_path, output_path, use_inertia_tensor=0, no_metrics=1):
    list_image = []
    input_map = get_input_map()
    cmap = get_cmap()
    for img in train_labels:
        img_T1 = sitk.ReadImage(input_path + "\\" + img)
        image_np_3d_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path + "\\" + img))

        first_vertebra = input("insert first vertebra name downwards for image " + img[:-7] + "    ")
        min_vertebra_value = input_map[first_vertebra.strip()]

        cc = sitk.ConnectedComponentImageFilter()
        erode = sitk.ErodeObjectMorphologyImageFilter()
        dilate = sitk.DilateObjectMorphologyImageFilter()
        erode.SetKernelRadius(2)
        dilate.SetKernelRadius(2)

        number_component = int(input("insert vertebrae number for image " + img[:-7] + "   "))

        image_np_3d = sitk.GetArrayFromImage(img_T1)
        regions_3d = regionprops(image_np_3d)
        mean = (regions_3d[0]['bbox'][2] + regions_3d[0]['bbox'][5]) // 2
        index_slice = mean - 1
        value_to_add = 1
        num = 0
        list_possibile_solution = []
        while 1:
            index_slice += value_to_add
            slice = image_np_3d[:, :, index_slice]
            slice = np.flip(slice, axis=0)

            image_sitk = sitk.GetImageFromArray(slice)
            image_erode = erode.Execute(image_sitk)
            image_dilate = dilate.Execute(image_erode)
            image_cc = cc.Execute(image_dilate)
            image_np = sitk.GetArrayFromImage(image_cc)
            regions = regionprops(image_np)

            properties = ['area', 'centroid', 'extent', 'perimeter', 'eccentricity', 'solidity']
            if use_inertia_tensor == 1:
                properties.append('inertia_tensor')
            regions_table = regionprops_table(image_np, properties=tuple(properties))

            arches = []
            bodies = []
            props = pandas.DataFrame(regions_table)

            if len(props) < number_component:
                if index_slice == regions_3d[0]['bbox'][5] - 1:
                    index_slice = mean
                    value_to_add = -1
                elif index_slice == regions_3d[0]['bbox'][2] + 1:
                    print('found ' + str(num) + ' results for image: ' + img)
                    break
                continue

            props = z_score_normalization(props)
            kmeans_obj = KMeans(n_clusters=2, random_state=46)
            kmeans_obj.fit(props)
            y_pred = kmeans_obj.predict(props)

            index, area_0, area_1 = 0, 0, 0

            for props in regions:
                if y_pred[index]:
                    area_1 += props['area']
                else:
                    area_0 += props['area']
                index += 1
            value = 1 if area_1 > area_0 else 0
            index = 0
            for props in regions:
                if y_pred[index] == value:
                    bodies.append(props)
                else:
                    arches.append(props)
                index += 1
            for arch in arches:
                if bodies:
                    min = [pow(pow(arch['centroid'][0] - bodies[0]['centroid'][0], 2) + pow(
                        arch['centroid'][1] - bodies[0]['centroid'][1], 2), 1 / 2), bodies[0]]
                    for body in bodies:
                        distance = pow(pow(arch['centroid'][0] - body['centroid'][0], 2) + pow(
                            arch['centroid'][1] - body['centroid'][1], 2), 1 / 2)
                        if distance < min[0]:
                            min = [distance, body]
                    image_np[image_np == arch['label']] = min[1]['label']

            label_value = 1
            for body in bodies:
                image_np[image_np == body['label']] = label_value
                label_value += 1

            centroids = [(image_np_3d.shape[0] - round(body['centroid'][0]),image_np_3d.shape[1], index_slice) for body in bodies]

            if len(bodies) == number_component:
                list_possibile_solution.append(centroids)
                plt.imshow(image_np, cmap)
                plt.axis("off")
                plt.title(str(num) + " " + img[:-7])
                plt.show()
                num += 1
            else:
                if index_slice == regions_3d[0]['bbox'][5] - 1:
                    index_slice = mean - 1
                    value_to_add = -1
                elif index_slice == regions_3d[0]['bbox'][2] + 1:
                    print('found ' + str(num) + ' results for image: ' + img)
                    break
        if num > 0:
            number_slice = int(input("insert selected image's number   "))
            nz = image_np_3d.nonzero()
            points_3d = [(x, y, z) for x, y, z in zip(*nz)]

            image_np_3d_new = np.zeros(shape=image_np_3d.shape)
            distance_3d = cdist(points_3d, list_possibile_solution[number_slice])
            min = np.argmin(distance_3d, axis=1)
            for i, point in enumerate(points_3d):
                image_np_3d_new[point] = min[i] + min_vertebra_value

            multi_dice, dices = multi_dice_coeff_with_list(image_np_3d_gt, image_np_3d_new,
                                                           number_component, min_vertebra_value)
            print("Multi-Dice: {:.4f} ".format(multi_dice))
            if no_metrics:
                multi_hausdorff = 0
                multi_assd = 0
            else:
                multi_hausdorff, hausdorff_distances = \
                    multi_hausdorff_distance_with_list(image_np_3d_gt,image_np_3d_new,
                                                       number_component, min_vertebra_value)
                print("Multi-Hausdorff Distance: {:.4f} ".format(multi_hausdorff))
                multi_assd, assd_distances = multi_assd_distance_with_list(image_np_3d_gt,
                                                                           image_np_3d_new,
                                                                           number_component,
                                                                           min_vertebra_value)
                print("Multi-ASSD: {:.4f} ".format(multi_assd))
            list_image.append((multi_dice, multi_hausdorff, multi_assd))

            image_3d = sitk.GetImageFromArray(image_np_3d_new)

            os.makedirs(output_path, exist_ok=True)

            sitk.WriteImage(image_3d, output_path + "\\" + img[:-7] + ".nii.gz")

            file = open(output_path + "\\" + img[:-7] + ".json", "w")
            file.write(json.dumps(centroids))
            file.close()
        else:
            list_image.append((0, 0, 0))
    del img_T1, image_np_3d_gt, image_np_3d
    list_image_np = np.array(list_image)
    list_mean = np.mean(list_image_np, axis=0)
    list_std = np.std(list_image_np, axis=0)
    print("Final Scores")
    print("Multi-Dice: {:.4f} +/- {:.4f}".format(list_mean, list_std))
    print("Multi-Hausdorff Distance: {:.4f} +/- {:.4f}".format(list_mean[1], list_std[1]))
    print("Multi-ASSD: {:.4f} +/- {:.4f}".format(list_mean[2], list_std[2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-class Vertebrae Segmentation")
    parser.add_argument(
        "-in",
        "--input-path",
        default=train_labels_folder, type=str,
        help="Specify the input folder path"
    )
    parser.add_argument(
        "-gt",
        "--gt-path",
        default=train_labels_folder, type=str,
        help="Specify the ground truth folder path"
    )
    parser.add_argument(
        "-out",
        "--output-path",
        default=multiclass_prediction_folder, type=str,
        help="Specify the output folder path"
    )
    parser.add_argument(
        "-ine",
        "--use-inertia-tensor",
        default=0, type=int,
        help="Specify if include inertia tensor for KMeans"
    )
    parser.add_argument(
        "-nom",
        "--no-metrics",
        default=1, type=int,
        help="Specify if calculating Hausdorff Distance and ASSD after multi-class Segmentation"
    )
    args = parser.parse_args()
    run(
        args.input_path,
        args.gt_path,
        args.output_path,
        args.use_inertia_tensor,
        args.no_metrics
    )
