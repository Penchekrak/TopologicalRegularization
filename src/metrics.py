import numpy as np
import torch
from scipy.linalg import sqrtm


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # # Numerical error might give slight imaginary component
    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         raise ValueError('Imaginary component {}'.format(m))
    #     covmean = covmean.real

    covmean = np.abs(covmean)

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def BW2_UVP_gaussian(sigma1, sigma2, entropy_weight, transport_map, original_sampler):
    sigma1_sqrt = torch.abs(torch.tensor(sqrtm(sigma1), dtype=torch.float32))
    D = torch.abs(torch.tensor(
        sqrtm(4 * sigma1_sqrt @ sigma2 @ sigma1_sqrt + entropy_weight ** 4 * torch.eye(sigma1.shape[0])),
        dtype=torch.float32
    ))
    C = 0.5 * sigma1_sqrt @ D @ torch.linalg.inv(sigma1_sqrt) - entropy_weight ** 2 / 2 * torch.eye(D.shape[0])
    cov_true_OT = torch.vstack([torch.hstack([sigma1, C]), torch.hstack([C.T, sigma2])])
    var_true_OT = np.trace(cov_true_OT)

    original_samples = original_sampler.sample(1000)
    generated_target_samples = transport_map.sample(1, context=original_samples).squeeze(1).detach().cpu().numpy()

    joined_samples = np.hstack([original_samples.cpu(), generated_target_samples])
    mu_generated = joined_samples.mean(0)
    cov_generated = np.cov(joined_samples.T)

    BW2_UVP = 100 * calculate_frechet_distance(
        np.zeros_like(mu_generated), cov_true_OT,
        mu_generated, cov_generated,
    ) / var_true_OT

    return BW2_UVP



