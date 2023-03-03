import abc
import typing as tp

import torch
from torch.utils.data import IterableDataset, Dataset

from torch import distributions as D


class _Sampler(IterableDataset):
    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()


Sampler = tp.Union[_Sampler, Dataset]


class ModulifiedDistribution(torch.nn.Module):
    def __setattr__(self, name: str, value: tp.Any) -> None:
        if type(value) is torch.Tensor:
            return super().register_buffer(name, value)
        return super().__setattr__(name, value)


class NormalSampler(ModulifiedDistribution, torch.distributions.Normal, _Sampler):
    def __init__(
            self,
            loc,
            scale=None
    ):
        ModulifiedDistribution.__init__(self)
        loc = torch.as_tensor(loc, dtype=torch.float32)
        if scale is not None:
            scale = torch.as_tensor(scale, dtype=torch.float32)
        else:
            scale = torch.ones_like(loc)
        torch.distributions.Normal.__init__(self, loc=loc, scale=scale, validate_args=False)


class DegenerateDistribution(_Sampler):
    def __init__(self):
        super().__init__()

    def sample(self, size: int = 1) -> torch.Tensor:
        return torch.zeros(1, 32, 32) + 0.5


class MixtureNormalSampler(_Sampler):
    def __init__(self, locs, scales=None, weights=None):
        locs = torch.tensor(locs, dtype=torch.float32)
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.float32)
        else:
            scales = torch.ones_like(locs)
        if weights is None:
            weights = torch.ones(locs.shape[0])
        normal = D.Independent(D.Normal(loc=locs, scale=scales), 1)
        selecting_distribution = D.Categorical(weights)
        self.mixture = D.MixtureSameFamily(selecting_distribution, normal)

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        return self.mixture.log_prob(samples)

    def sample(self) -> torch.Tensor:
        return self.mixture.sample()

# def symmetrize(X):
#     return np.real((X + X.T) / 2)
#
#
# class RandomUniformCovarianceNormal(Sampler):
#     def __init__(self, seed: int = 42):
#         pass


# class MoonSampler(Sampler):
#     def __init__(
#             self, scale=1.,
#     ):
#         super(MoonSampler, self).__init__()
#         self.scale = scale
#
#     def sample(self, batch_size=10):
#         batch = torch.tensor(datasets.make_moons(batch_size, noise=.1)[0], dtype=torch.float32)
#         batch *= self.scale
#         return batch.to(device)
#
#
# class SwissRollSampler(Sampler):
#     def __init__(
#             self, dim=2, scale=7.5
#     ):
#         super(SwissRollSampler, self).__init__()
#         assert dim == 2
#         self.dim = 2
#         self.scale = scale
#
#     def sample(self, batch_size=10):
#         batch = datasets.make_swiss_roll(
#             n_samples=batch_size,
#             noise=0.8
#         )[0].astype('float32')[:, [0, 2]] / self.scale
#         return torch.tensor(batch).to(device)
#
#
# class MixNGaussiansSampler(Sampler):
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

# def log_prob(self, x):
#     for center in self.centers:
#         normal = torch.distributions.Normal(loc=torch.tensor(center), scale=self.std)
#         nor
