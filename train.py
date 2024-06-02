import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

from src.trainable import Trainable
from src.utils import create_log_friendly_config_representation


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: DictConfig) -> None:
    logger = WandbLogger(**config['logger'],
                         # name=hydra.core.hydra_config.HydraConfig.get().job.config_name,
                         config=create_log_friendly_config_representation(config),
                         )
    trainer = Trainer(**hydra.utils.instantiate(config['trainer']), logger=logger, enable_checkpointing=False)
    trainable: Trainable = hydra.utils.instantiate(config['trainable'])
    datamodule = hydra.utils.instantiate(config['datamodule'])
    trainer.fit(trainable, datamodule=datamodule)


if __name__ == "__main__":
    main()
