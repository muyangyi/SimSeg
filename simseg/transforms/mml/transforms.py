import torchvision.transforms as transforms

from simseg.utils.registry import Registry
from simseg.transforms.mml.auto_augment import ImageNetPolicy
from simseg.transforms.mml.random_erasing import RandomErasing
from simseg.transforms.mml.gaussian_blur import ImgPilGaussianBlur
from simseg.transforms.mml.color_distortion import ImgPilColorDistortion

from simseg.utils import logger


TRANSFORMS = Registry('TRANSFORMS')

@TRANSFORMS.register_obj
def resize(cfg, **kwargs):
    size = cfg.transforms.resize.size
    return transforms.Resize((size,size))

@TRANSFORMS.register_obj
def resize_bicubic(cfg, **kwargs):
    size = cfg.transforms.resize_bicubic.size
    return transforms.Resize(size, interpolation=3)

@TRANSFORMS.register_obj
def center_crop(cfg, **kwargs):
    size = cfg.transforms.center_crop.size
    return transforms.CenterCrop(size)

@TRANSFORMS.register_obj
def random_crop(cfg, **kwargs):
    size = cfg.transforms.random_crop.size
    return transforms.RandomCrop(
        size
    )

@TRANSFORMS.register_obj
def random_flip(cfg, **kwargs):
    return transforms.RandomHorizontalFlip(p=0.5)

@TRANSFORMS.register_obj
def normalize(cfg, **kwargs):
    return transforms.Normalize(
        mean=cfg.transforms.normalize.mean, std=cfg.transforms.normalize.std
    )

@TRANSFORMS.register_obj
def autoaug(cfg, **kwargs):
    return ImageNetPolicy()

@TRANSFORMS.register_obj
def random_resize_crop(cfg, **kwargs):
    size = cfg.transforms.random_resize_crop.size
    scale = cfg.transforms.random_resize_crop.scale
    return transforms.RandomResizedCrop(size, scale=scale)

@TRANSFORMS.register_obj
def random_erasing(cfg, **kwargs):
    return RandomErasing(cfg.transforms.random_erasing.reprob, mode=cfg.transforms.random_erasing.remode, 
                         max_count=cfg.transforms.random_erasing.recount, num_splits=False, device='cpu')

@TRANSFORMS.register_obj
def color_distortion(cfg, **kwargs):
    return ImgPilColorDistortion(cfg.transforms.color_distortion.strength)

@TRANSFORMS.register_obj
def gaussian_blur(cfg, **kwargs):
    return ImgPilGaussianBlur(cfg.transforms.gaussian_blur.p, cfg.transforms.gaussian_blur.radius_min, cfg.transforms.gaussian_blur.radius_max)

@TRANSFORMS.register_obj
def color_jitter(cfg, **kwargs):
    color_jitter = (float(cfg.transforms.color_jitter),) * 3
    return transforms.ColorJitter(*color_jitter)

def build_transforms(cfg, mode="train"):
    transform_list = []

    transform_ops = (
        cfg.transforms.train_transforms
        if mode == "train"
        else cfg.transforms.valid_transforms
    )
    for tran in transform_ops:
        transform_list.append(TRANSFORMS.get(tran)(cfg))

    normalize = TRANSFORMS.get("normalize")(cfg)
    transform_list.extend([transforms.ToTensor(), normalize])
    if cfg.transforms.random_erasing.reprob > 0 and mode == 'train':
        transform_list.append(TRANSFORMS.get("random_erasing")(cfg))
    transform = transforms.Compose(transform_list)

    logger.emph(f'{mode} image transform is composed of:')
    logger.emph(transform)

    return transform