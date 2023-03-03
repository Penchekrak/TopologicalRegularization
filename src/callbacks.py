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
