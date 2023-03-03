import typing as tp

import pytorch_lightning as pl
import torch.optim
from omegaconf import OmegaConf
from torch.nn import Module, ModuleList

import src.utils
from src import utils
from src.losses import NormalizedMTopDivYXLoss


class Trainable(pl.LightningModule):
    def __init__(
            self,
            generator: Module,
            losses: tp.List[Module],
            optimizer_config: dict
    ):
        super(Trainable, self).__init__()
        self.generator = generator
        self.losses = ModuleList(losses)
        self.optimizer_config = optimizer_config

    def on_train_start(self) -> None:
        torch.set_float32_matmul_precision('medium')

    def on_train_epoch_start(self) -> None:
        if any(isinstance(loss, NormalizedMTopDivYXLoss) for loss in self.losses):
            iterator = iter(self.trainer.train_dataloader)
            batch1 = next(iterator).to(self.device)
            batch2 = next(iterator).to(self.device)[:batch1.shape[0] // 4]
            for loss in self.losses:
                if isinstance(loss, NormalizedMTopDivYXLoss):
                    loss.set_normalizing_constant(batch2, batch1)

    def training_step(
            self,
            batch: src.utils.Batch,
            batch_idx: int,
            optimizer_idx: int = 0
    ):
        log = {'trainer/global_step': self.trainer.global_step}
        optimization_mode = utils.OptimizationMode(optimizer_idx)
        prefix = ''
        if optimization_mode == utils.OptimizationMode.GENERATOR:
            output = self.generator(batch)
            prefix = 'Generator: '
        elif optimization_mode == utils.OptimizationMode.DISCRIMINATOR:
            with torch.no_grad():
                output = self.generator(batch)
            prefix = 'Discriminator: '

        loss = 0
        for loss_fn in self.losses:
            loss_value = loss_fn(batch, output, optimization_mode)
            if loss_value is not None:
                log[prefix + loss_fn._get_name()] = loss_value
                loss_value *= loss_fn.weight
                log[prefix + loss_fn._get_name() + ' weighted'] = loss_value
                loss += loss_value
        log[prefix + 'TotalLoss'] = loss

        self.trainer.logger.experiment.log(log)
        return loss

    @torch.no_grad()
    def validation_step(
            self,
            batch: src.utils.Batch,
            batch_idx: int
    ):
        # log = {'trainer/global_step': self.trainer.global_step}
        # prefix = ''
        output = self.generator(batch)
        # prefix = 'Validation: '
        # loss = 0
        # for loss_fn in self.losses:
        #     loss_value = loss_fn(batch, output, optimization_mode)
        #     if loss_value is not None:
        #         log[prefix + loss_fn._get_name()] = loss_value
        #         loss_value *= loss_fn.weight
        #         log[prefix + loss_fn._get_name() + ' weighted'] = loss_value
        #         loss += loss_value
        # log[prefix + 'TotalLoss'] = loss

        # self.trainer.logger.experiment.log(log)
        return output

    @staticmethod
    def build_lr_scheduler_from_config(
            config: dict,
            optimizer: torch.optim.Optimizer
    ):
        config['scheduler'] = config['scheduler'](optimizer=optimizer)
        return config

    @staticmethod
    def build_optimizer_from_config(
            config: dict,
            parameters: tp.Iterator[torch.nn.Parameter]
    ):
        config['optimizer'] = config['optimizer'](params=parameters)
        if 'lr_scheduler' in config:
            config['lr_scheduler'] = Trainable.build_lr_scheduler_from_config(
                config['lr_scheduler'],
                optimizer=config['optimizer']
            )
        return config

    def configure_optimizers(self):
        generator_optimizer = Trainable.build_optimizer_from_config(
            OmegaConf.to_container(self.optimizer_config['generator']),
            self.generator.parameters()
        )
        has_trainable_losses = any(p.requires_grad for p in self.losses.parameters())
        if has_trainable_losses:
            losses_optimizer = Trainable.build_optimizer_from_config(
                OmegaConf.to_container(self.optimizer_config['losses']),
                self.losses.parameters()
            )
            return [
                generator_optimizer,
                losses_optimizer
            ]
        else:
            return generator_optimizer
