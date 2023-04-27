import gc
import typing as tp

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import wandb
from einops import einops
from matplotlib import collections as mc
from matplotlib.axes import Axes
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
import torch
from torch import linalg

import src.utils
from src.trainable import Trainable


import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
from matplotlib import pyplot as plt
import pickle as pkl


class LogLikelihood(Callback):
    ll_sum: float
    ll_terms_count: int

    @staticmethod
    def log_likelihood(logp):
        return logp.sum()

    @torch.no_grad()
    def on_validation_batch_end(
            self,
            trainer: Trainer,
            trainable: Trainable,
            outputs: tp.Any,
            batch: src.utils.Batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.ll_sum += self.log_likelihood(
            trainer.datamodule.sampler.log_prob(
                outputs.cpu()
            )
        ).item()
        self.ll_terms_count += outputs.shape[0]

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            trainable: Trainable
    ) -> None:
        trainer.logger.experiment.log(
            {
                'LogLikelihood': self.ll_sum / self.ll_terms_count,
                "trainer/global_step": trainer.global_step
            }
        )

    def on_validation_epoch_start(
            self,
            trainer: Trainer,
            trainable: Trainable
    ) -> None:
        self.ll_sum = 0.0
        self.ll_terms_count = 0


class PlotDistributions2D(Callback):
    target_samples: torch.Tensor
    generated_samples: torch.Tensor

    class PotentialData(tp.NamedTuple):
        x_grid: np.ndarray
        y_grid: np.ndarray
        z_grid: np.ndarray

    def __init__(
            self,
            add_potential_contour: bool = False,
            potential_contour_resolution: int = 512,
            potential_level_counts: int = 20,
            # number of points to display on distributions plot
            data_samples_count: int = 256
    ):
        self.add_potential_contour = add_potential_contour
        self.potential_contour_resolution = potential_contour_resolution
        self.potential_levels_count = potential_level_counts
        self.data_samples_count = data_samples_count

    def on_validation_epoch_start(
            self,
            trainer: Trainer,
            trainable: Trainable
    ) -> None:
        self.target_samples = torch.empty(0, 2)
        self.generated_samples = torch.empty(0, 2)

    def plot_generated_2D(
            self,
            target_samples: np.ndarray,
            generated_samples: np.ndarray,
            potential_data_extractor: tp.Optional[tp.Callable[..., PotentialData]]
    ):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharex=True, sharey=True, dpi=150)

        for axis in axes:
            axis.grid(True)
            axis.set_aspect('equal')

        axes[0].scatter(target_samples[:, 0], target_samples[:, 1], c='peru', edgecolors='black')
        axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], c='wheat',
                        edgecolors='black')

        if potential_data_extractor is not None:
            axes[1].contour(
                *potential_data_extractor(*axes[1].get_xlim(), *axes[1].get_ylim()),
                levels=self.potential_levels_count,
                cmap='magma'
            )
        else:
            axes[1].scatter(target_samples[:, 0], target_samples[:, 1], c='peru', edgecolors='black', alpha=0.2)

        axes[0].set_title(r'Target $y\sim\mathbb{Q}$', fontsize=22, pad=10)
        axes[1].set_title(r'Fitted $G(z)_{\#}\mathbb{P}$', fontsize=22, pad=10)

        return fig

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            trainable: Trainable,
            outputs: tp.Any,
            batch: src.utils.Batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        target_samples = batch
        generated_samples = outputs
        num_generated_samples_to_take = min(self.data_samples_count - len(self.generated_samples),
                                            len(generated_samples))
        if num_generated_samples_to_take > 0:
            self.generated_samples = torch.cat((
                self.generated_samples,
                generated_samples[:num_generated_samples_to_take].cpu()
            ), dim=0)

        num_target_samples_to_take = min(self.data_samples_count - len(self.target_samples), len(target_samples))
        if num_target_samples_to_take > 0:
            self.target_samples = torch.cat((
                self.target_samples,
                target_samples[:num_target_samples_to_take].cpu()
            ), dim=0)

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            trainable: Trainable
    ) -> None:
        target_samples_numpy = self.target_samples.numpy()
        generated_samples_numpy = self.generated_samples.numpy()
        assert \
            (target_samples_numpy.shape[-1] == 2) and \
            (generated_samples_numpy.shape[-1] == 2), \
            'Distribution plotting is supported only for 2d data'
        if self.add_potential_contour:
            potential_data_extractor = self.build_potential_data_exptractor(trainable.potential)
        else:
            potential_data_extractor = None

        distributions_plot_image = self.plot_generated_2D(target_samples_numpy,
                                                          generated_samples_numpy,
                                                          potential_data_extractor=potential_data_extractor)
        trainer.logger.experiment.log(
            {
                'distributions': wandb.Image(distributions_plot_image),
                "trainer/global_step": trainer.global_step
            }
        )
        plt.close(distributions_plot_image)

    def build_potential_data_exptractor(self, potential):
        def potential_data_extractor(xmax, xmin, ymax, ymin):
            device = next(iter(potential.parameters())).device
            x = torch.linspace(xmin, xmax, steps=self.potential_contour_resolution, device=device)
            y = torch.linspace(ymin, ymax, steps=self.potential_contour_resolution, device=device)
            x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
            xy = torch.stack((x_grid, y_grid), dim=-1)
            xy_batched = einops.rearrange(xy, 'x_size y_size spatial_dim -> (x_size y_size) spatial_dim')
            potential_values = potential(xy_batched)
            potential_values = einops.rearrange(
                potential_values, '(x_size y_size) 1 -> x_size y_size',
                x_size=self.potential_contour_resolution,
                y_size=self.potential_contour_resolution
            )
            potential_data = PlotDistributions2D.PotentialData(
                x_grid.cpu().numpy(),
                y_grid.cpu().numpy(),
                potential_values.cpu().numpy()
            )
            return potential_data

        return potential_data_extractor


class PlotSampledImages(Callback):
    target_samples: tp.Union[torch.Tensor, tp.List]
    generated_samples: tp.Union[torch.Tensor, tp.List]
    current_samples_count: int

    def __init__(
            self,
            samples_count: int = 8,
    ):
        self.samples_count = samples_count

    def on_validation_epoch_start(self, trainer, trainable: Trainable) -> None:
        self.current_samples_count = 0
        self.generated_samples = []
        self.target_samples = []

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            trainable: Trainable,
            outputs: tp.Any,
            batch: src.utils.Batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        num_samples_to_take = min(self.samples_count - self.current_samples_count, len(batch))
        if num_samples_to_take > 0:
            self.target_samples.append(
                batch[:num_samples_to_take].cpu()
            )
            self.generated_samples.append(
                outputs[:num_samples_to_take].cpu()
            )
            self.current_samples_count += num_samples_to_take

    def on_validation_epoch_end(
            self,
            trainer: Trainer,
            trainable: Trainable
    ) -> None:
        target_samples = torch.cat(self.target_samples, dim=0).numpy()
        generated_samples_numpy = torch.cat(self.generated_samples, dim=0).numpy()

        source_samples_grid = einops.rearrange(
            target_samples,
            'b c h w -> h (b w) c'
        )

        generated_samples_grid = einops.rearrange(
            generated_samples_numpy,
            'b c h w -> h (b w) c'
        )

        sampled_images_fig, (source_samples_axis, latent_samples_axes) = \
            plt.subplots(2, 1,
                         figsize=(self.samples_count * 3.5, 2 * 3.5),
                         subplot_kw={'aspect': 'auto'},
                         gridspec_kw={'left': 0.2, 'width_ratios': [1], 'height_ratios': [1, 1]}
                         )

        source_samples_axis.imshow(source_samples_grid)
        source_samples_axis.get_xaxis().set_visible(False)
        source_samples_axis.set_yticks([])
        source_samples_axis.set_ylabel(r'$x\sim\mathbb{Q}$', fontsize=30)

        latent_samples_axes.imshow(generated_samples_grid)

        latent_samples_axes.get_xaxis().set_visible(False)
        latent_samples_axes.set_yticks([])
        latent_samples_axes.set_ylabel(r'$G(z)#\mathbb{P}$ for different $z$', fontsize=30)

        sampled_images_fig.tight_layout(pad=0.001)
        trainer.logger.experiment.log(
            {
                'sampled images': wandb.Image(sampled_images_fig),
                "trainer/global_step": trainer.global_step
            }
        )
        plt.close(sampled_images_fig)


class GC(Callback):
    def __init__(self, clean_every_n_epochs: int = 1):
        self.clean_every_n_epochs = clean_every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, trainable: Trainable) -> None:
        if (trainer.current_epoch + 1) % self.clean_every_n_epochs == 0:
            gc.collect()
            torch.cuda.empty_cache()

class NDBCallback(Callback):
    def __init__(
            self,
            samples_count: int = 100,
            number_of_bins=10,
            significance_level=0.05,
            z_threshold=None,
            whitening=False,
            max_dims=None,
    ):
        """
        NDB Evaluation Class
        :param number_of_bins: Number of bins (clusters) default=100
        :param significance_level: The statistical significance level for the two-sample test
        :param z_threshold: Allow defining a threshold in terms of difference/SE for defining a bin as statistically different
        :param whitening: Perform data whitening - subtract mean and divide by per-dimension std
        :param max_dims: Max dimensions to use in K-means. By default derived automatically from d
        """
        self.samples_count = samples_count
        self.number_of_bins = number_of_bins
        self.significance_level = significance_level
        self.z_threshold = z_threshold
        self.whitening = whitening
        self.ndb_eps = 1e-6
        self.training_mean = 0.0
        self.training_std = 1.0
        self.max_dims = max_dims
        self.bin_centers = None
        self.bin_proportions = None
        self.ref_sample_size = None
        self.used_d_indices = None

    def on_validation_epoch_start(self, trainer, trainable: Trainable) -> None:
        self.current_samples_count = 0
        self.generated_samples = []
        self.target_samples = []

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            trainable: Trainable,
            outputs: tp.Any,
            batch: src.utils.Batch,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        num_samples_to_take = min(self.samples_count - self.current_samples_count, len(batch))
        if num_samples_to_take > 0:
            self.target_samples.append(
                batch[:num_samples_to_take].cpu()
            )
            self.generated_samples.append(
                outputs[:num_samples_to_take].cpu()
            )
            self.current_samples_count += num_samples_to_take

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_samples = torch.cat(self.target_samples, dim=0).numpy()
        generated_samples_numpy = torch.cat(self.generated_samples, dim=0).numpy()
        self.construct_bins(target_samples)
        result = self.evaluate(generated_samples_numpy)
        trainer.logger.experiment.log({
            **result,
            "trainer/global_step": trainer.global_step
        })

    def construct_bins(self, training_samples):
        """
        Performs K-means clustering of the training samples
        :param training_samples: An array of m x d floats (m samples of dimension d)
        """

        n, d = training_samples.shape
        k = self.number_of_bins
        if self.whitening:
            self.training_mean = np.mean(training_samples, axis=0)
            self.training_std = np.std(training_samples, axis=0) + self.ndb_eps

        if self.max_dims is None and d > 1000:
            # To ran faster, perform binning on sampled data dimension (i.e. don't use all channels of all pixels)
            self.max_dims = d//6

        whitened_samples = (training_samples-self.training_mean)/self.training_std
        d_used = d if self.max_dims is None else min(d, self.max_dims)
        self.used_d_indices = np.random.choice(d, d_used, replace=False)

        print('Performing K-Means clustering of {} samples in dimension {} / {} to {} clusters ...'.format(n, d_used, d, k))
        print('Can take a couple of minutes...')
        if n//k > 1000:
            print('Training data size should be ~500 times the number of bins (for reasonable speed and accuracy)')

        clusters = KMeans(n_clusters=k, max_iter=100, n_jobs=-1).fit(whitened_samples[:, self.used_d_indices])

        bin_centers = np.zeros([k, d])
        for i in range(k):
            bin_centers[i, :] = np.mean(whitened_samples[clusters.labels_ == i, :], axis=0)

        # Organize bins by size
        label_vals, label_counts = np.unique(clusters.labels_, return_counts=True)
        bin_order = np.argsort(-label_counts)
        self.bin_proportions = label_counts[bin_order] / np.sum(label_counts)
        self.bin_centers = bin_centers[bin_order, :]
        self.ref_sample_size = n
        print('Done.')

    def evaluate(self, query_samples):
        """
        Assign each sample to the nearest bin center (in L2). Pre-whiten if required. and calculate the NDB
        (Number of statistically Different Bins) and JS divergence scores.
        :param query_samples: An array of m x d floats (m samples of dimension d)
        :param model_label: optional label string for the evaluated model, allows plotting results of multiple models
        :return: results dictionary containing NDB and JS scores and array of labels (assigned bin for each query sample)
        """
        n = query_samples.shape[0]
        query_bin_proportions, query_bin_assignments = self.__calculate_bin_proportions(query_samples)
        # print(query_bin_proportions)
        different_bins = NDBCallback.two_proportions_z_test(self.bin_proportions, self.ref_sample_size, query_bin_proportions,
                                                    n, significance_level=self.significance_level,
                                                    z_threshold=self.z_threshold)
        ndb = np.count_nonzero(different_bins)
        js = NDBCallback.jensen_shannon_divergence(self.bin_proportions, query_bin_proportions)
        results = {f'{self.number_of_bins}_NDB': ndb,
                   f'{self.number_of_bins}_JS': js,
                   # 'Proportions': query_bin_proportions,
                   # 'N': n,
                   # 'Bin-Assignment': query_bin_assignments,
                   f'{self.number_of_bins}_Different-Bins': different_bins}


        print('NDB =', ndb, 'NDB/K =', ndb/self.number_of_bins, ', JS =', js)
        return results



    def __calculate_bin_proportions(self, samples):
        if self.bin_centers is None:
            print('First run construct_bins on samples from the reference training data')
        assert samples.shape[1] == self.bin_centers.shape[1]
        n, d = samples.shape
        k = self.bin_centers.shape[0]
        D = np.zeros([n, k], dtype=samples.dtype)

        print('Calculating bin assignments for {} samples...'.format(n))
        whitened_samples = (samples-self.training_mean)/self.training_std
        for i in range(k):
            print('.', end='', flush=True)
            D[:, i] = np.linalg.norm(whitened_samples[:, self.used_d_indices] - self.bin_centers[i, self.used_d_indices],
                                     ord=2, axis=1)
        print()
        labels = np.argmin(D, axis=1)
        probs = np.zeros([k])
        label_vals, label_counts = np.unique(labels, return_counts=True)
        probs[label_vals] = label_counts / n
        return probs, labels


    @staticmethod
    def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
        # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
        # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        # Allow defining a threshold in terms as Z (difference relative to the SE) rather than in p-values.
        if z_threshold is not None:
            return abs(z) > z_threshold
        p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))    # Two-tailed test
        return p_values < significance_level

    @staticmethod
    def jensen_shannon_divergence(p, q):
        """
        Calculates the symmetric Jensen–Shannon divergence between the two PDFs
        """
        m = (p + q) * 0.5
        return 0.5 * (NDBCallback.kl_divergence(p, m) + NDBCallback.kl_divergence(q, m))

    @staticmethod
    def kl_divergence(p, q):
        """
        The Kullback–Leibler divergence.
        Defined only if q != 0 whenever p != 0.
        """
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(q))
        assert not np.any(np.logical_and(p != 0, q == 0))

        p_pos = (p > 0)
        return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))




if __name__ == "__main__":
    dim=100
    k=100
    n_train = k*100
    n_test = k*10

    train_samples = np.random.uniform(size=[n_train, dim])
    ndb = NDB(training_data=train_samples, number_of_bins=k, whitening=True)

    test_samples = np.random.uniform(high=1.0, size=[n_test, dim])
    ndb.evaluate(test_samples, model_label='Test')

    test_samples = np.random.uniform(high=0.9, size=[n_test, dim])
    ndb.evaluate(test_samples, model_label='Good')

    test_samples = np.random.uniform(high=0.75, size=[n_test, dim])
    ndb.evaluate(test_samples, model_label='Bad')

    ndb.plot_results(models_to_plot=['Test', 'Good', 'Bad'])