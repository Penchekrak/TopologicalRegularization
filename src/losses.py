import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from src import utils
from src.utils import Batch, compute_cross_barcodes, get_random_sample
import typing as tp


class Loss(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.register_buffer('weight', torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch: Batch, output: tp.Any, optimization_mode: utils.OptimizationMode,
                ) -> torch.Tensor:
        """
        :param output:
        :param batch:
        :return:
        """
        raise NotImplementedError()


class StandardGANLoss(Loss):
    def __init__(self, discriminator, weight: float = 1.0):
        super().__init__(weight)
        self.discriminator = discriminator
        self.criterion = nn.BCELoss()

    def forward(self, batch: Batch, output: tp.Any, optimization_mode: utils.OptimizationMode) -> torch.Tensor:
        if optimization_mode == utils.OptimizationMode.GENERATOR:
            return self.criterion(
                self.discriminator(output),
                output.new_ones(output.shape[0], 1)
            )
        elif optimization_mode == utils.OptimizationMode.DISCRIMINATOR:
            return \
                    self.criterion(
                        self.discriminator(output),
                        output.new_zeros(output.shape[0], 1)
                    ) + \
                    self.criterion(
                        self.discriminator(batch),
                        batch.new_ones(batch.shape[0], 1)
                    )


class WassersteinGANLoss(Loss):
    def __init__(self, discriminator, weight: float = 1.0):
        super().__init__(weight)
        self.discriminator = discriminator

    def forward(self, batch: Batch, output: tp.Any, optimization_mode: utils.OptimizationMode,
                ) -> torch.Tensor:
        if optimization_mode == utils.OptimizationMode.GENERATOR:
            return - self.discriminator(output).mean()
        elif optimization_mode == utils.OptimizationMode.DISCRIMINATOR:
            return self.discriminator(output).mean() - self.discriminator(batch).mean()


class WassersteinGPGANLoss(WassersteinGANLoss):
    def __init__(self, gp_lambda=10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gp_lambda = gp_lambda

    def calculate_gradient_penalty(
            self,
            real,
            fake
    ):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((real.shape[0], *(1 for _ in range(real.ndim - 1))), device=real.device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

        model_interpolates = self.discriminator(interpolates)
        grad_outputs = torch.ones_like(model_interpolates, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    def forward(self, batch: Batch, output: tp.Any, optimization_mode: utils.OptimizationMode) -> torch.Tensor:
        if optimization_mode == utils.OptimizationMode.GENERATOR:
            return - self.discriminator(output).mean()
        elif optimization_mode == utils.OptimizationMode.DISCRIMINATOR:
            return self.discriminator(output).mean() - self.discriminator(batch).mean() + \
                self.gp_lambda * self.calculate_gradient_penalty(
                    batch,
                    output
                )


def mtopdiv(points_a, points_b, dim):
    homologies = compute_cross_barcodes(points_a, points_b, dim=dim + 1)
    barcode = homologies[0][dim]
    return torch.sum(barcode[:, 1] - barcode[:, 0])


class MTopDivYXLoss(Loss):
    def __init__(self, dimension: int = 0, weight: float = 1.0, denoising_fraction: int = 4):
        super().__init__(weight)
        self.dim = dimension
        self.denoising_fraction = denoising_fraction

    def _forward_impl(self, batch: Batch, output: tp.Any) -> torch.Tensor:
        return mtopdiv(get_random_sample(batch, batch.shape[0] // self.denoising_fraction), output, self.dim)

    def forward(self, batch: Batch, output: tp.Any, optimization_mode: utils.OptimizationMode) -> torch.Tensor:
        if optimization_mode == utils.OptimizationMode.GENERATOR:
            return self._forward_impl(batch, output)
        elif optimization_mode == utils.OptimizationMode.DISCRIMINATOR:
            return None


class MTopDivXYLoss(MTopDivYXLoss):
    def _forward_impl(self, batch: Batch, output: tp.Any) -> torch.Tensor:
        return super()._forward_impl(output, batch)


class NormalizedMTopDivYXLoss(MTopDivYXLoss):
    normalizing_constant: torch.Tensor

    @torch.no_grad()
    def set_normalizing_constant(self, batch1, batch2):
        self.normalizing_constant = mtopdiv(batch1, batch2, self.dim)

    def _forward_impl(self, batch: Batch, output: tp.Any) -> torch.Tensor:
        return super()._forward_impl(batch, output) / self.normalizing_constant


class NormalizedMTopDivXYLoss(NormalizedMTopDivYXLoss, MTopDivXYLoss):
    pass


class SquaredNormalizedMTopDivYXLoss(NormalizedMTopDivYXLoss):
    def _forward_impl(self, batch: Batch, output: tp.Any) -> torch.Tensor:
        return (super()._forward_impl(batch, output) - 1) ** 2


class SquaredNormalizedMTopDivXYLoss(SquaredNormalizedMTopDivYXLoss, NormalizedMTopDivXYLoss):
    pass


class PerceptualDistanceSquaredNormalizedMTopDivYXLoss(SquaredNormalizedMTopDivYXLoss):
    def __init__(self, weights_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet18()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    def _forward_impl(self, batch: Batch, output: tp.Any):
        x_vector, y_vector = self.get_feature_vectors(batch, output)
        return super()._forward_impl(y_vector, x_vector)

    @torch.no_grad()
    def set_normalizing_constant(self, batch1, batch2):
        x_vector, y_vector = self.get_feature_vectors(batch1, batch2)
        super().set_normalizing_constant(x_vector, y_vector)

    def get_feature_vectors(self, batch, output):
        output = (output - self.mean) / self.std
        batch = (batch - self.mean) / self.std
        if self.resize:
            output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)
            batch = self.transform(batch, mode='bilinear', size=(224, 224), align_corners=False)
        x = output
        x_vector = x.new_empty(x.shape[0], 0)
        for block in self.blocks:
            x = block(x)
            x_vector = torch.cat((x_vector, x.flatten(1)), dim=-1)
        with torch.no_grad():
            y = batch
            y_vector = y.new_empty(y.shape[0], 0)
            for block in self.blocks:
                y = block(y)
                y_vector = torch.cat((y_vector, y.flatten(1)), dim=-1)
        return x_vector, y_vector
