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


# from src.models.flows.transforms import Dequantize
#
#
# class DequantizeTransform(Dequantize):
#     def forward(
#             self,
#             inputs: torch.Tensor,
#             context: tp.Optional[torch.Tensor] = None
#     ):
#         outputs, _ = super().forward(inputs, context)
#         return outputs
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
        # if isinstance(target_type, list):
        #     self.target_type = target_type
        # else:
        #     self.target_type = [target_type]

        # if not self.target_type and self.target_transform is not None:
        #     raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        # identity = self._load_csv("identity_CelebA.txt")
        # bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        # landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        # attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        # self.identity = identity.data[mask]
        # self.bbox = bbox.data[mask]
        # self.landmarks_align = landmarks_align.data[mask]
        # self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        # self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        # self.attr_names = attr.header

    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        if self.transform is not None:
            X = self.transform(X)
        return X

    def __len__(self) -> int:
        return len(self.filename)


# USPS = make_dataset_image_only(_USPS)
# def load_dataset(name, path, img_size=64, batch_size=64, test_ratio=0.1):
#     if name == 'mnist':
#         mnist_transforms = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor()
#         ])
#         dataset = MNIST(path, train=True, transform=mnist_transforms, download=True)
#     elif name == 'usps':
#         mnist_transforms = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor()
#         ])
#         dataset = USPS(path, train=True, transform=mnist_transforms, download=True)
#     elif name in ['shoes', 'handbag', 'outdoor', 'church']:
#         dataset = h5py_to_dataset(path, img_size)
#     elif name in ['celeba_female', 'celeba_male', 'aligned_anime_faces', 'describable_textures']:
#         transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset = ImageFolder(path, transform=transform)
#     else:
#         raise Exception('Unknown dataset')
#
#     if name in ['celeba_female', 'celeba_male']:
#         with open('../datasets/list_attr_celeba.txt', 'r') as f:
#             lines = f.readlines()[2:]
#         if name == 'celeba_female':
#             idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
#         else:
#             idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] != '-1']
#     elif dataset == 'describable_textures':
#         idx = np.random.RandomState(seed=0xBAD5EED).permutation(len(dataset))
#     else:
#         idx = list(range(len(dataset)))
#
#     test_size = int(len(idx) * test_ratio)
#     train_idx, test_idx = idx[:-test_size], idx[-test_size:]
#     train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
#
#     train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=0, batch_size=batch_size))
#     test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=0, batch_size=batch_size))
#     return train_sampler, test_sampler

#     def __init__(self, dim=2, N=8, with_central=False, std=1, r=12):
#         super(MixNGaussiansSampler, self).__init__()
#         assert dim == 2
#         assert N <= 8
#         self.dim = 2
#         self.std, self.r = std, r
#
#         self.with_central = with_central
#         centers = [
#             (1, 0), (-1, 0), (0, 1), (0, -1),
#             (1. / np.sqrt(2), 1. / np.sqrt(2)),
#             (1. / np.sqrt(2), -1. / np.sqrt(2)),
#             (-1. / np.sqrt(2), 1. / np.sqrt(2)),
#             (-1. / np.sqrt(2), -1. / np.sqrt(2))
#         ]
#         if self.with_central:
#             centers.append((0, 0))
#         self.centers = torch.tensor(centers[:N], dtype=torch.float32)
#
#     def sample(self, batch_size=10):
#         with torch.no_grad():
#             batch = torch.randn(batch_size, self.dim)
#             indices = random.choices(range(len(self.centers)), k=batch_size)
#             batch *= self.std
#             batch += self.r * self.centers[indices, :]
#         return batch.to(device)

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
