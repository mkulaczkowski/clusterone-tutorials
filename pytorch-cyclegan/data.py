import glob
import os

import torch.utils.data as data
from torchvision import transforms
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageDataset(data.Dataset):
    def __init__(self, root, transform=None, loader=pil_loader, extensions=IMG_EXTENSIONS):
        image_paths = [path for path in glob.glob(os.path.join(root, '*')) if
                       any(path.lower().endswith(ext) for ext in extensions)]
        if len(image_paths) == 0:
            raise RuntimeError('Found 0 files in directory: ' + root + '\n'
                               'Supported extensions are: ' + ','.join(extensions))

        self.root = root
        self.extensions = extensions
        self.loader = loader
        self.image_paths = image_paths

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_transforms(opts):
    transform_list = [transforms.Resize((286, 286), Image.BICUBIC)]

    if opts.mode == 'train':
        transform_list += [transforms.RandomCrop((256, 256)),
                           transforms.RandomHorizontalFlip()]
    else:
        transform_list += [transforms.CenterCrop((256, 256))]

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def get_inference_transform():
    transform_list = [transforms.Resize((256, 256), Image.BICUBIC),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_reverse_transform():
    transform_list = [transforms.Normalize((-1., -1., -1.),
                                           (2.0, 2.0, 2.0)),
                      transforms.ToPILImage()]
    return transforms.Compose(transform_list)
