import os
import sys

logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)

base_dataset_dir = 'F:\\Dataset\\Verse'
train_images_folder = os.path.join(base_dataset_dir, "training")
train_labels_folder = os.path.join(base_dataset_dir, "training_mask")

os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)

original_train_images_folder = os.path.join(base_dataset_dir, "original_training")
original_train_labels_folder = os.path.join(base_dataset_dir, "original_training_mask")
train_prediction_folder = os.path.join(base_dataset_dir, "predTr")
multiclass_prediction_folder = os.path.join(base_dataset_dir, "predMulticlass")
train_images = os.listdir(train_images_folder)
train_labels = os.listdir(train_labels_folder)
original_train_images = os.listdir(original_train_images_folder)
original_train_labels = os.listdir(original_train_labels_folder)

train_images = [train_image for train_image in train_images
                if train_image.endswith(".nii.gz") and not train_image.startswith('.')]
train_labels = [train_label for train_label in train_labels
                if train_label.endswith(".nii.gz") and not train_label.startswith('.')]
original_train_images = [train_image for train_image in original_train_images
                if train_image.endswith(".nii.gz") and not train_image.startswith('.')]
original_train_labels = [train_label for train_label in original_train_labels
                if train_label.endswith(".nii.gz") and not train_label.startswith('.')]

test_images_folder = os.path.join(base_dataset_dir, "validation")
os.makedirs(test_images_folder, exist_ok=True)
test_images = os.listdir(test_images_folder)
test_prediction_folder = os.path.join(base_dataset_dir, "predTs")

test_images = [test_image for test_image in test_images
               if test_image.endswith(".nii.gz") and not test_image.startswith('.')]

labels_names = {
   "0": "background",
   "1": "Spine",
 }
labels_names_list = [labels_names[el] for el in labels_names]
