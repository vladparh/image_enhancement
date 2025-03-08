import torch
from enhancer import Enhancer
from model.NAFNet_arch import NAFNetLocal
from PIL import Image
from torchvision.transforms import ToTensor, v2
from torchvision.transforms.functional import to_pil_image


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img1 = Image.open(
        "C:/Users/Vlad/Desktop/ВКР/datasets/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene084/blur/blur_2.png"
    )
    img2 = Image.open(
        "C:/Users/Vlad/Desktop/ВКР/datasets/RealBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene101/blur/blur_2.png"
    )
    img1 = img1.convert("RGB")
    img1 = ToTensor()(img1)
    img1 = v2.functional.center_crop(img1, [500, 600]).unsqueeze(0).to(device)
    img2 = img2.convert("RGB")
    img2 = ToTensor()(img2)
    img2 = v2.functional.center_crop(img2, [500, 600]).unsqueeze(0).to(device)
    img = torch.concat((img1, img2), dim=0)
    model = NAFNetLocal(
        width=32,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )
    model.load_state_dict(
        torch.load(
            "C:/Users/Vlad/Desktop/ВКР/image_enchancement/src/naf_net/weights/NAFNet-GoPro-width32.pth"
        )["params"]
    )
    enhancer = Enhancer(model=model)
    enhancer = enhancer.to(device)
    img = enhancer.predict(img, use_split=True)
    img1 = to_pil_image(img[0].squeeze(0).clamp(0, 1))
    img2 = to_pil_image(img[1].squeeze(0).clamp(0, 1))
    img1.show()
    img2.show()


if __name__ == "__main__":
    main()
