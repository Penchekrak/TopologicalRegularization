import typing as tp
from enum import Enum

import torch
from einops import einops
from omegaconf import OmegaConf, DictConfig
from torchph.pershom.pershom_backend import vr_persistence


def recursively_rebuild_dict_with_transformations(
        reference_dict: tp.Dict[str, tp.Any],
        transformations_by_key: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        transformations_by_type: tp.Optional[tp.Dict[tp.Type, tp.Callable]] = None,
        filter_keys: tp.Optional[tp.Set[tp.Tuple[str, ...]]] = None,
        current_key_prefix: tp.Tuple[str, ...] = tuple()
):
    if transformations_by_key is None:
        transformations_by_key = {}
    if transformations_by_type is None:
        transformations_by_type = {}
    if filter_keys is None:
        filter_keys = set()
    generated_dict = {}
    for key in reference_dict:
        composite_key = current_key_prefix + (key,)
        if composite_key in filter_keys:
            continue
        elif composite_key in transformations_by_key:
            op = transformations_by_key[key]
        elif isinstance(reference_dict[key], dict):
            op = recursively_rebuild_dict_with_transformations
        elif type(reference_dict[key]) in transformations_by_type:
            op = transformations_by_type[type(reference_dict[key])]
        else:
            op = lambda obj, *args, **kwargs: obj
        generated_dict[key] = op(
            reference_dict[key],
            transformations_by_key=transformations_by_key,
            transformations_by_type=transformations_by_type,
            filter_keys=filter_keys,
            current_key_prefix=composite_key
        )
    return generated_dict


def list_of_targets_to_dict(
        lst,
        transformations_by_key: tp.Optional[tp.Dict[str, tp.Callable]] = None,
        transformations_by_type: tp.Optional[tp.Dict[tp.Type, tp.Callable]] = None,
        filter_keys: tp.Optional[tp.Set[tp.Tuple[str, ...]]] = None,
        current_key_prefix: tp.Tuple[str, ...] = tuple()
):
    if all(isinstance(entry, dict) and '_target_' in entry for entry in lst):
        generated_dict = {}
        for entry in lst:
            key = entry['_target_']
            del entry['_target_']
            entry['used'] = True  # dummy key to display in wandb and indicate usage of particular callback or loss
            generated_dict[key] = recursively_rebuild_dict_with_transformations(
                entry,
                transformations_by_key=transformations_by_key,
                transformations_by_type=transformations_by_type,
                filter_keys=filter_keys,
                current_key_prefix=current_key_prefix + (key,)
            )
        return generated_dict
    else:
        return lst


def create_log_friendly_config_representation(config: DictConfig):
    raw_dict = OmegaConf.to_container(config, resolve=True)
    proper_dict = recursively_rebuild_dict_with_transformations(
        raw_dict, {},
        filter_keys={('logger',)},
        transformations_by_type={list: list_of_targets_to_dict}
    )
    return proper_dict


def batch_and_latent_to_batch(tensor: torch.Tensor):
    return einops.rearrange(tensor, 'batch latent ... -> (batch latent) ...')


Batch = tp.Any


class OptimizationMode(Enum):
    GENERATOR = 0
    DISCRIMINATOR = 1


def conform_dimensions(latent_samples, source_samples, target_samples=None):
    batch_size, n_latent_samples, *latent_space_dim = latent_samples.size()
    batch_size, *source_space_dim = source_samples.size()
    latent_samples = latent_samples.view(batch_size * n_latent_samples, *latent_space_dim)
    source_samples = source_samples \
        .unsqueeze(1) \
        .expand(batch_size, n_latent_samples, *source_space_dim) \
        .reshape(batch_size * n_latent_samples, *source_space_dim)
    if target_samples is not None:
        batch_size, *target_space_dim = target_samples.size()
        target_samples = target_samples \
            .unsqueeze(1) \
            .expand(batch_size, n_latent_samples, *target_space_dim) \
            .reshape(batch_size * n_latent_samples, *target_space_dim)
        return latent_samples, source_samples, target_samples
    return latent_samples, source_samples


def turn_scheduler_on(num_step):
    def lambda_lr(step):
        return min(1., step / num_step)

    return lambda_lr


def compute_cross_barcodes(x, y, dim):
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    dist_x_x = torch.cdist(x, x)
    dist_x_y = torch.cdist(x, y)
    dummy = torch.zeros(y.shape[0], y.shape[0], device=y.device)
    distance_matrix = torch.cat((torch.cat((dist_x_x, dist_x_y), 1), torch.cat((dist_x_y.T, dummy), 1)), 0)
    homologies = vr_persistence(distance_matrix, max_dimension=dim)
    return homologies


def get_random_sample(tensor, sample_size):
    if sample_size == tensor.shape[0]:
        return tensor
    elif sample_size < tensor.shape[0]:
        return tensor[torch.randperm(tensor.shape[0], device=tensor.device)[:sample_size]]
    else:
        raise ValueError("Sample size should be less than tensor leading dimension")
