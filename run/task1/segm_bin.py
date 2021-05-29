import argparse
import os
import torch
import torchio
from config.config_semseg import SemSegMRIConfig
from config.paths import logs_folder, test_prediction_folder, test_images_folder
from run.utils import print_test, nii_load, nii_write
from models.vnet3d import VNet3D


def run(path_image_in, path_mask_out, use_temp_net=False, temp_epoch=100):
    ##########################
    # Config
    ##########################
    config = SemSegMRIConfig()

    ###########################
    # Load Net
    ###########################
    cuda_dev = torch.device("cuda")

    if use_temp_net:
        path_net = os.path.join(logs_folder, "model_epoch_{:04d}.pht".format(temp_epoch))
    else:
        path_net = os.path.join(logs_folder, "model.pt")

    ###########################
    # Eval Loop
    ###########################
    os.makedirs(path_mask_out, exist_ok=True)
    test_images_in = os.listdir(path_image_in)
    print("Images Folder: {}".format(path_image_in))

    print_test()
    if use_temp_net:
        net = VNet3D(num_outs=config.num_outs, channels=config.num_channels)
        net.load_state_dict(torch.load(path_net))
    else:
        net = torch.load(path_net)
    net = net.to(cuda_dev)
    net.eval()

    for idx, test_image in enumerate(test_images_in):
        print("Iter {} on {}".format(idx + 1, len(test_images_in)))
        print("Image: {}".format(test_image))

        train_image_path = os.path.join(path_image_in, test_image)
        train_image_np, affine = nii_load(train_image_path)
        grid_sampler = torchio.inference.GridSampler(
            torchio.Subject(t1=torchio.Image(type=torchio.INTENSITY, path=train_image_path)),
            64,
            4,
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=config.batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler)

        with torch.no_grad():
            for patches_batch in patch_loader:
                inputs = patches_batch['t1'][torchio.DATA].to(cuda_dev)
                locations = patches_batch[torchio.LOCATION]
                labels = net(inputs.float())
                labels = labels.argmax(dim=torchio.CHANNELS_DIMENSION, keepdim=True)
                aggregator.add_batch(labels, locations)

        outputs_np = aggregator.get_output_tensor().numpy()
        outputs_np = outputs_np[0]  # Z x Y x X
        original_shape = train_image_np.shape
        outputs_np = outputs_np[:original_shape[0], :original_shape[1], :original_shape[2]]

        filename_out = os.path.join(path_mask_out, test_image)
        nii_write(outputs_np, affine, filename_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Binary Segmentation")
    parser.add_argument(
        "-in",
        "--path_image_in",
        default=test_images_folder, type=str,
        help="Specify the input folder path"
    )
    parser.add_argument(
        "-out",
        "--path_mask_out",
        default=test_prediction_folder, type=str,
        help="Specify the output folder path"
    )
    args = parser.parse_args()
    run(args.path_image_in, args.path_mask_out, use_temp_net=True)
