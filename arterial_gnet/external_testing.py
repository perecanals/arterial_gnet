
import os
import argparse

from arterial_gnet.dataloading.data_augmentation import get_transforms
from arterial_gnet.dataloading.dataloading import get_test_data_loader
from arterial_gnet.test.test import run_external_testing

import torch

def main(model_dir, root_test, dataset_name, device=0):

    model_name = os.path.basename(model_dir)

    print("------------------------------------------------")
    print("Running external testing for model:", model_name)

    # Read device
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    #################################### Dataset organization ############################################
    # Define pre-transforms (applied to the graph before batching, regardless of training or testing)
    args_ = argparse.Namespace(radius=int(model_name.split("_r-")[1].split("_")[0]), concat=False)
    _, _, test_transform = get_transforms(device, args_)
    # Get data loaders
    test_loader = get_test_data_loader(root_test, None, test_transform, radius=args_.radius)

    model_name = os.path.basename(model_dir)

    # Run testing
    for fold in range(5):
        for model in ["latest"]:
            run_external_testing(
                model_dir,
                model_name,
                test_loader,
                dataset_name=dataset_name,
                model = model,
                device = device,
                fold = fold,
                is_classification = True
            )

if __name__ == "__main__":
    import os

    root_test = "/media/Disk_B/databases/ArterialMaps/root_test"
    dataset_name = "test_dataset_2023"

    model_name = "ArterialGNet_bs-64_te-500_hc-32_hcd-32_op-adam_lr-0.001_lrs-poly_ngl-0_nsl-0_ndl-1_nol-1_ah-4_agg-mean_drop-0.2_r-10_concat-True_wl-exp_os-False_rs-400_trs-44_class_tag-prova"
    root = "/media/Disk_B/databases/ArterialMaps/root"
    model_dir = os.path.join(root, "models", model_name)

    import time

    start = time.time()

    main(model_dir, root_test, dataset_name)

    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end - start))