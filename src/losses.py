import typing as tp

import torch
import torchvision
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn

from src import utils
from src.utils import Batch, compute_cross_barcodes, get_random_sample


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


class RandomDistanceMatrixLoss(Loss):
    def __init__(self, weight: float = 1.0, fraction=0.01):
        super().__init__(weight)
        self.fraction = fraction

    def _forward_impl(self, batch, output):
        if self.fraction is None:
            n_take = batch.shape[0] - 1
        else:
            n_take = int(batch.shape[0] * self.fraction)
        pairs = get_random_sample(
            torch.cartesian_prod(
                torch.arange(batch.shape[0]),
                torch.arange(output.shape[0]),
            ),
            n_take
        )
        random_dists = torch.dist(
            batch[pairs[:, 0]],
            output[pairs[:, 1]],
        )

        return random_dists.sum()

    def forward(self, batch: Batch, output: tp.Any, optimization_mode: utils.OptimizationMode) -> torch.Tensor:
        if optimization_mode == utils.OptimizationMode.GENERATOR:
            return self._forward_impl(batch, output)
        elif optimization_mode == utils.OptimizationMode.DISCRIMINATOR:
            return None


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.cuda()
        for p in self.model.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').unsqueeze(-1).unsqueeze(-1)

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
        # if self.resize:
        #     output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)
        #     batch = self.transform(batch, mode='bilinear', size=(224, 224), align_corners=False)
        x = output
        x_vector = self.calc_embs(x)
        with torch.no_grad():
            y = batch
            y_vector = self.calc_embs(y)
        return x_vector, y_vector

    def calc_embs(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        return x


class PerceptualMTopDivYXLoss(MTopDivYXLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.cuda()
        for p in self.model.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').unsqueeze(-1).unsqueeze(-1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').unsqueeze(-1).unsqueeze(-1)

    def _forward_impl(self, batch: Batch, output: tp.Any):
        x_vector, y_vector = self.get_feature_vectors(batch, output)
        return super()._forward_impl(y_vector, x_vector)

    def get_feature_vectors(self, batch, output):
        output = (output - self.mean) / self.std
        batch = (batch - self.mean) / self.std
        # if self.resize:
        #     output = self.transform(output, mode='bilinear', size=(224, 224), align_corners=False)
        #     batch = self.transform(batch, mode='bilinear', size=(224, 224), align_corners=False)
        x = output
        x_vector = self.calc_embs(x)
        with torch.no_grad():
            y = batch
            y_vector = self.calc_embs(y)
        return x_vector, y_vector

    def calc_embs(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        return x



class PerceptualArcFaceMTopDivYXLoss(MTopDivYXLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
        for p in self.model.parameters():
            p.requires_grad = False

    def _forward_impl(self, batch: Batch, output: tp.Any):
        x_vector, y_vector = self.get_feature_vectors(batch, output)
        return super()._forward_impl(y_vector, x_vector)


    def calc_embs(self, x):
        return self.model(torch.nn.functional.interpolate(x, (160, 160)))

    def get_feature_vectors(self, batch, output):
        x = output
        x_vector = self.calc_embs(x)
        with torch.no_grad():
            y = batch
            y_vector = self.calc_embs(y)
        return x_vector, y_vector
