import pytest
import torch
import math
from src.losses import compute_cross_barcodes, mtopdiv
from src.losses import (
    MTopDivYXLoss,
    MTopDivXYLoss,
    NormalizedMTopDivYXLoss,
    NormalizedMTopDivXYLoss,
    SquaredNormalizedMTopDivXYLoss,
    SquaredNormalizedMTopDivYXLoss,
    PerceptualDistanceSquaredNormalizedMTopDivYXLoss,
)


def sort(tensor):
    sorted, indices = torch.sort(tensor)
    return sorted


@pytest.mark.parametrize(
    ['points_a', 'points_b', 'answer'],
    [
        (
                torch.FloatTensor([  # points_a
                    [-1, -1],
                    [-1, 1]
                ]),
                torch.FloatTensor([  # points_b
                    [1, -1],
                    [1, 1]
                ]),
                torch.FloatTensor([
                    2.0,  # side of the square formed by points_a
                    2.0  # side of the square formed by closest points from a and b
                ])
        ),
        # (
        #         torch.FloatTensor([  # points_a
        #             [-1, -1],
        #             [-1, 1],
        #             [0.5, 0.5]
        #         ]),
        #         torch.FloatTensor([  # same point cloud
        #             [-1, -1],
        #             [-1, 1],
        #             [0.5, 0.5]
        #         ]),
        #         torch.FloatTensor([
        #         ])
        # ),
    ]
)
def test_compute_zeroth_cross_barcodes(points_a, points_b, answer):
    homologies = compute_cross_barcodes(points_a.cuda(), points_b.cuda(), dim=2)
    zeroth_barcode = homologies[0][0]
    assert torch.allclose(
        zeroth_barcode[:, 0],
        torch.zeros_like(zeroth_barcode[:, 0])  # birth times for zeroth homologies is always 0
    )
    assert torch.allclose(
        torch.sort(zeroth_barcode[:, 1])[0],  # death times
        torch.sort(answer)[0].cuda()
    )


@pytest.mark.parametrize(
    ['xy_criterion', 'yx_criterion'],
    [
        (MTopDivXYLoss(denoising_fraction=1), MTopDivYXLoss(denoising_fraction=1)),
        (NormalizedMTopDivXYLoss(denoising_fraction=1), NormalizedMTopDivYXLoss(denoising_fraction=1)),
        (SquaredNormalizedMTopDivXYLoss(denoising_fraction=1), SquaredNormalizedMTopDivYXLoss(denoising_fraction=1))
    ]
)
def test_flip_losses(xy_criterion, yx_criterion):
    xy_criterion.normalizing_constant = 1.0
    yx_criterion.normalizing_constant = 1.0
    for _ in range(100):
        points_a = torch.randn(10, 10, device='cuda')
        points_b = torch.randn(10, 10, device='cuda')
        print('losses', xy_criterion._forward_impl(points_a, points_b),
              yx_criterion._forward_impl(points_b, points_a))
        assert torch.allclose(
            xy_criterion._forward_impl(points_a, points_b),
            yx_criterion._forward_impl(points_b, points_a)
        )
