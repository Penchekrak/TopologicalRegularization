import gc
import typing as tp
from types import SimpleNamespace

import numpy as np
import torch
import wandb
from einops import einops
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans

import src.utils
from src.models.lenet import LeNet5
from src.trainable import Trainable
from facenet_pytorch import InceptionResnetV1


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
        source_samples_axis.set_ylabel('Target', fontsize=30)

        latent_samples_axes.imshow(generated_samples_grid)

        latent_samples_axes.get_xaxis().set_visible(False)
        latent_samples_axes.set_yticks([])
        latent_samples_axes.set_ylabel('Generated', fontsize=30)

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

    def on_validation_epoch_end(self, trainer: Trainer, trainable: Trainable) -> None:
        self.on_train_epoch_end(trainer, trainable)

mnist_classifier = LeNet5().cuda()
facenet = InceptionResnetV1(pretrained='vggface2').eval().cuda()


feature_extractors = dict(
    flattener=lambda x: x.flatten(start_dim=1, end_dim=-1),
    mnist_extractor=mnist_classifier.compute_embedding,
    face_extractor=lambda x: facenet(torch.nn.functional.interpolate(x, (160, 160))),
)


class NDBCallback(Callback):
    target_samples: tp.Union[torch.Tensor, tp.List]
    generated_samples: tp.Union[torch.Tensor, tp.List]
    current_samples_count: int

    def __init__(
            self,
            samples_count: int = 8,
            number_of_bins=100,
            significance_level=0.05,
            z_threshold=None,
            whitening=False,
            max_dims=None,
            feature_extractor=None
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
        self.feature_extractor = feature_extractors.get(feature_extractor, lambda x: x)

    def on_validation_epoch_start(self, trainer: Trainer, trainable: Trainable) -> None:
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
                self.feature_extractor(batch[:num_samples_to_take]).cpu()
            )
            self.generated_samples.append(
                self.feature_extractor(outputs[:num_samples_to_take]).cpu()
            )
            self.current_samples_count += num_samples_to_take

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Trainable) -> None:
        self.target_samples = torch.cat(self.target_samples).numpy()
        self.generated_samples = torch.cat(self.generated_samples).numpy()
        generated_bins, target_bins = self.compute_histograms(reference=self.target_samples,
                                                              query=self.generated_samples)
        results = self.evaluate(target_bins, generated_bins)
        trainer.logger.experiment.log(
            results |
            {
                "trainer/global_step": trainer.global_step
            }
        )

    @staticmethod
    def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):
        """Computes the PRD curve for discrete distributions.

        This function computes the PRD curve for the discrete distribution eval_dist
        with respect to the reference distribution ref_dist. This implements the
        algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
        equiangular grid of num_angles values between [0, pi/2].

        Args:
          eval_dist: 1D NumPy array or list of floats with the probabilities of the
                     different states under the distribution to be evaluated.
          ref_dist: 1D NumPy array or list of floats with the probabilities of the
                    different states under the reference distribution.
          num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                      The default value is 1001.
          epsilon: Angle for PRD computation in the edge cases 0 and pi/2. The PRD
                   will be computes for epsilon and pi/2-epsilon, respectively.
                   The default value is 1e-10.

        Returns:
          precision: NumPy array of shape [num_angles] with the precision for the
                     different ratios.
          recall: NumPy array of shape [num_angles] with the recall for the different
                  ratios.

        Raises:
          ValueError: If not 0 < epsilon <= 0.1.
          ValueError: If num_angles < 3.
        """

        if not (epsilon > 0 and epsilon < 0.1):
            raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
        if not (num_angles >= 3 and num_angles <= 1e6):
            raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

        # Compute slopes for linearly spaced angles between [0, pi/2]
        angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=num_angles)
        slopes = np.tan(angles)

        # Broadcast slopes so that second dimension will be states of the distribution
        slopes_2d = np.expand_dims(slopes, 1)

        # Broadcast distributions so that first dimension represents the angles
        ref_dist_2d = np.expand_dims(ref_dist, 0)
        eval_dist_2d = np.expand_dims(eval_dist, 0)

        # Compute precision and recall for all angles in one step via broadcasting
        precision = np.minimum(ref_dist_2d * slopes_2d, eval_dist_2d).sum(axis=1)
        recall = precision / slopes

        # handle numerical instabilities leaing to precision/recall just above 1
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)

        return precision, recall

    @staticmethod
    def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
        """Computes F_beta scores for the given precision/recall values.

        The F_beta scores for all precision/recall pairs will be computed and
        returned.

        For precision p and recall r, the F_beta score is defined as:
        F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)

        Args:
          precision: 1D NumPy array of precision values in [0, 1].
          recall: 1D NumPy array of precision values in [0, 1].
          beta: Beta parameter. Must be positive. The default value is 1.
          epsilon: Small constant to avoid numerical instability caused by division
                   by 0 when precision and recall are close to zero.

        Returns:
          NumPy array of same shape as precision and recall with the F_beta scores for
          each pair of precision/recall.

        Raises:
          ValueError: If any value in precision or recall is outside of [0, 1].
          ValueError: If beta is not positive.
        """

        if not ((precision >= 0).all() and (precision <= 1).all()):
            raise ValueError('All values in precision must be in [0, 1].')
        if not ((recall >= 0).all() and (recall <= 1).all()):
            raise ValueError('All values in recall must be in [0, 1].')
        if beta <= 0:
            raise ValueError('Given parameter beta %s must be positive.' % str(beta))

        return (1 + beta ** 2) * (precision * recall) / (
                (beta ** 2 * precision) + recall + epsilon)

    def evaluate(self, reference_histogram, query_histogram):
        """
        Assign each sample to the nearest bin center (in L2). Pre-whiten if required. and calculate the NDB
        (Number of statistically Different Bins) and JS divergence scores.
        :param query_samples: An array of m x d floats (m samples of dimension d)
        :param model_label: optional label string for the evaluated model, allows plotting results of multiple models
        :return: results dictionary containing NDB and JS scores and array of labels (assigned bin for each query sample)
        """
        different_bins = NDBCallback.two_proportions_z_test(reference_histogram,
                                                            self.target_samples.shape[0],
                                                            query_histogram,
                                                            self.generated_samples.shape[0],
                                                            significance_level=self.significance_level,
                                                            z_threshold=self.z_threshold)
        ndb = np.count_nonzero(different_bins)
        tvd = 0.5 * np.sum(np.abs(reference_histogram - query_histogram))
        js = NDBCallback.jensen_shannon_divergence(reference_histogram, query_histogram)
        precision, recall = self.compute_prd(query_histogram, reference_histogram)
        plot = self.plot_precision_recall_curve(precision, recall)
        results = {'NDB ratio': ndb / self.number_of_bins,
                   'JS': js,
                   'precision recall': wandb.Image(plot),
                   'f1 max': np.max(self._prd_to_f_beta(precision, recall)),
                   'total variation distance': tvd,
                   }
        if self.target_samples.shape[1] == 2:
            results |= {
                'clustering': wandb.Image(self.plot_clusterization())
            }
        return results

    @staticmethod
    def two_proportions_z_test(p1, n1, p2, n2, significance_level, z_threshold=None):
        # Per http://stattrek.com/hypothesis-test/difference-in-proportions.aspx
        # See also http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/binotest.htm
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        # Allow defining a threshold in terms as Z (difference relative to the SE) rather than in p-values.
        if z_threshold is not None:
            return abs(z) > z_threshold
        p_values = 2.0 * norm.cdf(-1.0 * np.abs(z))  # Two-tailed test
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

    def compute_histograms(self, reference, query):
        if self.whitening:
            training_mean = np.mean(reference, axis=0)
            training_std = np.std(reference, axis=0) + self.ndb_eps
            reference = (reference - training_mean) / training_std
        self.kmeans = MiniBatchKMeans(n_clusters=self.number_of_bins, n_init=10, compute_labels=True)
        reference_labels = self.kmeans.fit(reference).labels_

        if self.whitening:
            query = (query - training_mean) / training_std
        eval_labels = self.kmeans.predict(query)

        eval_bins = np.histogram(eval_labels, bins=self.number_of_bins,
                                 range=[0, self.number_of_bins], density=True)[0]
        ref_bins = np.histogram(reference_labels, bins=self.number_of_bins,
                                range=[0, self.number_of_bins], density=True)[0]
        return eval_bins, ref_bins

    @staticmethod
    def plot_precision_recall_curve(precision, recall):
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Recall (coverage)', fontsize=12)
        ax.set_ylabel('Precision (quality)', fontsize=12)
        ax.set_aspect('equal')
        fig.tight_layout()
        return fig

    def plot_clusterization(self):
        from scipy.spatial import Voronoi, voronoi_plot_2d

        centers = self.kmeans.cluster_centers_
        fig, ax = plt.subplots()
        ax.scatter(centers[:, 0], centers[:, 1], marker='s', c='red', label='Cluster centers')
        ax.scatter(self.generated_samples[:, 0], self.generated_samples[:, 1], alpha=0.5, c='orange',
                   label='Generated samples')
        ax.scatter(self.target_samples[:, 0], self.target_samples[:, 1], alpha=0.5, c='green', label='Target samples')

        vor = Voronoi(centers)
        fig = voronoi_plot_2d(vor, ax, show_vertices=False, show_points=False)
        ax.legend(loc='best')
        ax.set_aspect('equal')
        fig.tight_layout(pad=0.001)
        return fig


class MNISTClassDistributionCallback(Callback):
    generated_samples: tp.Union[torch.Tensor, tp.List]
    current_samples_count: int

    def __init__(
            self,
            samples_count,
    ):
        from torch.distributions import Categorical
        self.compute_entropy = lambda logits_tensor: Categorical(logits=logits_tensor).entropy().sum()
        self.samples_count = samples_count
        self.classifier = mnist_classifier

    def on_validation_epoch_start(
            self,
            trainer: Trainer,
            trainable: Trainable
    ) -> None:
        self.class_labels = []
        self.sum_entropies = 0
        self.current_samples_count = 0

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
        num_samples_to_take = min(self.samples_count - self.current_samples_count, len(batch))
        if num_samples_to_take > 0:
            logits = self.classifier(outputs[:num_samples_to_take])
            self.sum_entropies += self.compute_entropy(logits)
            self.class_labels += torch.argmax(logits, dim=-1).cpu().tolist()
            self.current_samples_count += num_samples_to_take

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Trainable) -> None:
        histogram = np.unique(self.class_labels, return_counts=True)
        fig, ax = plt.subplots()
        probs = histogram[1] / self.current_samples_count
        kolmogorov_dist = np.max(np.abs(probs - 1 / 10))
        ax.bar([str(i) for i in histogram[0]], probs)
        ax.set_title("Histogram of generated class distribution")
        trainer.logger.experiment.log(
            {
                "class distribution": wandb.Image(fig),
                "average entropy": self.sum_entropies / self.current_samples_count,
                "kolmogorov distance": kolmogorov_dist,
                "trainer/global_step": trainer.global_step
            }
        )


class FIDCallback(Callback):
    def __init__(self):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.fid = FrechetInceptionDistance(feature=768)

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
        self.fid.update(batch, real=True)
        self.fid.update(outputs, real=False)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: Trainable) -> None:
        trainer.logger.experiment.log(
            {
                "FID": self.fid.compute(),
                "trainer/global_step": trainer.global_step
            }
        )


if __name__ == "__main__":
    dim = 2
    k = 100
    n_train = k * 100
    n_test = k * 10

    train_samples = np.random.uniform(size=[n_train, dim])
    ndb = NDBCallback(samples_count=100, whitening=True)

    ndb.target_samples = [torch.randn(size=[n_test, dim])]
    ndb.generated_samples = [torch.randn(size=[n_test, dim]) + 2]

    res_dct = None


    class logger:
        def log(self, dct):
            global res_dct
            res_dct = dct


    trainer = SimpleNamespace(logger=SimpleNamespace(experiment=logger()), global_step=1)

    ndb.on_validation_epoch_end(trainer, None)
    print(res_dct)
    res_dct['precision recall'].show()
    res_dct['clustering'].show()
