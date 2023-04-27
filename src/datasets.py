import os
import typing as tp

import PIL
import torch
import torchvision.datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST as _MNIST, USPS as _USPS, CelebA as _CelebA, VisionDataset
from torchvision.datasets.utils import verify_str_arg


#
def make_dataset_image_only(dataset_class: tp.Type[torchvision.datasets.VisionDataset]):
    def __init__(
            self,
            transform: tp.List[tp.Callable],
            *args,
            **kwargs
    ):
        if not any(isinstance(t, transforms.ToTensor) for t in transform):
            optional_to_tensor = [transforms.ToTensor()]
        else:
            optional_to_tensor = []
        transform = transforms.Compose([
            *transform,
            *optional_to_tensor,
        ])
        dataset_class.__init__(self, transform=transform, *args, **kwargs)

    def __getitem__(self, item):
        img, target = dataset_class.__getitem__(self, item)
        return img

    image_only_dataset = type(dataset_class.__class__.__name__, (dataset_class,),
                              {'__init__': __init__, '__getitem__': __getitem__})
    return image_only_dataset


MNIST = make_dataset_image_only(_MNIST)


class CelebA(_CelebA):
    def __init__(
            self,
            root: str,
            split: str = "train",
            # target_type="attr",
            transform=None,
            target_transform=None,
            download: bool = False,
    ) -> None:
        VisionDataset.__init__(self, root, transform=transform, target_transform=target_transform)
        self.split = split
        if download:
            self.download()

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        if self.transform is not None:
            X = self.transform(X)
        return X

    def __len__(self) -> int:
        return len(self.filename)


class SingleSamplerDataModule(LightningDataModule):
    def __init__(self, sampler, num_workers: int = 8, batch_size: int = 32):
        super().__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.sampler, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.sampler, num_workers=self.num_workers, batch_size=self.batch_size)


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data", num_workers: int = 8, batch_size: int = 32, image_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = image_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(
            root=self.data_dir,
            transform=[
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ],
            train=False,
            download=False)
        self.mnist_train = MNIST(
            root=self.data_dir,
            transform=[
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ],
            train=True,
            download=False
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, num_workers=self.num_workers, batch_size=self.batch_size)


class CelebADataModule(LightningDataModule):
    def __init__(self, data_dir: str = "data", num_workers: int = 8, batch_size: int = 32, image_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = image_size

    def setup(self, stage: str):
        self.celeba_test = CelebA(
            root=self.data_dir,
            transform=transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ]),
            split='valid',
            download=False)
        self.celeba_train = CelebA(
            root=self.data_dir,
            transform=transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
            ]),
            split='train',
            download=False
        )

    def train_dataloader(self):
        return DataLoader(self.celeba_train, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.celeba_test, num_workers=self.num_workers, batch_size=self.batch_size)
