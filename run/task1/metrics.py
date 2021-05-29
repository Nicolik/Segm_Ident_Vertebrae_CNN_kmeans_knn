import os
import numpy as np
import SimpleITK as sitk
import json
import medpy.metric as mm
from tqdm import tqdm

from config.paths import base_dataset_dir


def run():
    path_to_ground_truth_masks = os.path.join(base_dataset_dir, 'validation_mask')
    path_to_predicted_masks = os.path.join(base_dataset_dir, 'predTs')

    gt_masks = os.listdir(path_to_ground_truth_masks)
    pr_masks = os.listdir(path_to_predicted_masks)

    assert len(gt_masks) == len(pr_masks), "Mismatch in validation sizes!"

    gt_masks.sort()
    pr_masks.sort()

    gt_masks_ = [g.split('_')[0] for g in gt_masks]
    pr_masks_ = [g.split('_')[0] for g in pr_masks]

    assert gt_masks_ == pr_masks_, "Validation masks are not the same!"

    dices = list()
    recalls = list()
    precisions = list()
    assds = list()
    hds = list()

    for i in tqdm(range(len(gt_masks))):
        filename_mask = gt_masks[i]
        pred_mask = pr_masks[i]
        path_to_gt_mask = os.path.join(path_to_ground_truth_masks, filename_mask)
        path_to_pr_mask = os.path.join(path_to_predicted_masks, pred_mask)
        gt_sitk = sitk.ReadImage(path_to_gt_mask)
        pr_sitk = sitk.ReadImage(path_to_pr_mask)
        gt_np = sitk.GetArrayFromImage(gt_sitk)
        pr_np = sitk.GetArrayFromImage(pr_sitk)

        if gt_np.shape != pr_np.shape:
            new_gt_np = np.zeros(pr_np.shape)
            z,y,x = gt_np.shape
            new_gt_np[:z,:y,:x] = gt_np
            gt_np = new_gt_np.copy()
            print("Zero-pad on Ground Truth {}".format(filename_mask))

        dice = mm.dc(pr_np, gt_np)
        dices.append(dice)
        recall = mm.recall(pr_np, gt_np)
        recalls.append(recall)
        precision = mm.precision(pr_np, gt_np)
        precisions.append(precision)
        assd = mm.assd(pr_np, gt_np)
        assds.append(assd)
        hd = mm.hd(pr_np, gt_np)
        hds.append(hd)

    dices_np = np.array(dices)
    recalls_np = np.array(recalls)
    precisions_np = np.array(precisions)
    assds_np = np.array(assds)
    hds_np = np.array(hds)

    print("Dice      = {:.2f} +/- {:.2f}".format(np.mean(dices_np)*100, np.std(dices_np)*100))
    print("Recall    = {:.2f} +/- {:.2f}".format(np.mean(recalls_np)*100, np.std(recalls_np)*100))
    print("Precision = {:.2f} +/- {:.2f}".format(np.mean(precisions_np)*100, np.std(precisions_np)*100))
    print("ASSD      = {:.2f} +/- {:.2f}".format(np.mean(assds_np), np.std(assds_np)))
    print("HD        = {:.2f} +/- {:.2f}".format(np.mean(hds_np), np.std(hds_np)))

    print("{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}\t{:.2f} ± {:.2f}".format(
        np.mean(dices_np)*100, np.std(dices_np)*100,
        np.mean(recalls_np)*100, np.std(recalls_np)*100,
        np.mean(precisions_np)*100, np.std(precisions_np)*100,
        np.mean(assds_np), np.std(assds_np),
        np.mean(hds_np), np.std(hds_np)
    ))

    dict_out = {
        'dices': dices,
        'recalls': recalls,
        'precisions': precisions,
        'assds': assds,
        'hds': hds
    }

    with open("metrics.json") as f:
        json.dump(dict_out, f)


if __name__ == '__main__':
    run()
