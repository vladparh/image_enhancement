import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.models.mlwnet.MLWNet_arch import MLWNet_Local
from src.models.naf_net.dataset import TestDataset
from src.models.naf_net.trainer import NetTrainer


def main():
    test_blur_img_paths = []
    test_gt_img_paths = []
    file = open(
        "C:/Users/Vlad/Desktop/ВКР/datasets/RealBlur/RealBlur_J_test_list.txt", "r"
    )
    for line in file.readlines():
        gt_img, blur_img = line.split()
        test_blur_img_paths.append(blur_img)
        test_gt_img_paths.append(gt_img)
    file.close()

    val_dataset = TestDataset(
        dataset_path="C:/Users/Vlad/Desktop/ВКР/datasets/RealBlur",
        blur_img_paths=test_blur_img_paths,
        gt_img_paths=test_gt_img_paths,
        is_crop=False,
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = MLWNet_Local(dim=32)
    model.load_state_dict(
        torch.load(
            "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/models/mlwnet/weights/realblur_j-width32.pth",
            weights_only=True,
        )["params"]
    )

    module = NetTrainer(model, lr=None, min_lr=None, n_epochs=None, loss_fn=None)
    trainer = pl.Trainer()
    trainer.validate(model=module, dataloaders=val_loader)


if __name__ == "__main__":
    main()
