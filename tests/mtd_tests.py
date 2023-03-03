import pytest
import torch
import math
from src.losses import compute_cross_barcodes, mtopdiv

def sort(tensor):
    sorted, indices = torch.sort(tensor)
    return sorted

@pytest.mark.parametrize(
    ['points_a', 'points_b', 'answer'],
    [
        (
                torch.FloatTensor([  # points_a
                    [-1, -1],
                    [-1,  1]
                ]),
                torch.FloatTensor([  # points_b
                    [1, -1],
                    [1,  1]
                ]),
                torch.FloatTensor([
                    2.0,  # side of the square formed by points_a
                    2.0  # side of the square formed by closest points from a and b
                ])
        ),
        (
                torch.FloatTensor([  # points_a
                    [-1, -1],
                    [-1,  1],
                    [ 0.5,  0.5]
                ]),
                torch.FloatTensor([  # same point cloud
                    [-1, -1],
                    [-1,  1],
                    [ 0.5,  0.5]
                ]),
                torch.FloatTensor([
                    0.0,  # all homologies die instantly
                    0.0
                ])
        )
    ]
)
def test_compute_zeroth_cross_barcodes(points_a, points_b, answer):
    homologies = compute_cross_barcodes(points_a.cuda(), points_b.cuda(), dim=2)
    zeroth_barcode = homologies[0][0]
    print(homologies)
    print(zeroth_barcode)
    assert torch.allclose(
        zeroth_barcode[:, 0],
        torch.zeros_like(zeroth_barcode[:, 0])  # birth times for zeroth homologies is always 0
    )
    assert torch.allclose(
        torch.sort(zeroth_barcode[:, 1])[0],  # death times
        torch.sort(answer)[0].cuda()
    )