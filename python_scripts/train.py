"""
Main script for training diner
"""

from omegaconf import OmegaConf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.data.pl_datamodule import PlDataModule
from pytorch_lightning import Trainer
from src.models.diner import DINER
from src.models.keypointnerf import KeypointNeRFLightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
from src.util.general import copy_python_files
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

def main():
    config_path = sys.argv[1]
    model_name = sys.argv[2]
    if model_name not in ['DINER', 'KeypointNeRF']:
        raise ValueError(f'Model Name should be DINER or KeypointNeRF but got: {model_name}')
    
    conf = OmegaConf.load(config_path)
    conf_logger = conf.logger_diner if model_name == 'DINER' else conf.logger_keypointnerf
    os.makedirs(conf_logger.kwargs.save_dir, exist_ok=True)
    datamodule = PlDataModule(conf.data.train, conf.data.val, model_name)
    datamodule.setup()

    # initialize model
    if model_name == 'DINER':
        model = DINER(nerf_conf=conf.nerf, renderer_conf=conf.renderer, znear=datamodule.train_set.znear,
                              zfar=datamodule.train_set.zfar, **conf.optimizer_diner.kwargs)
    else:
        model = KeypointNeRFLightningModule(conf.keypoint_nerf, **conf.optimizer_keypointnerf.kwargs)

    # initialize logger
    logger = TensorBoardLogger(**conf_logger.kwargs, name=None)

    # save configuration
    os.makedirs(logger.log_dir, exist_ok=True)
    os.system(f"cp {config_path} {os.path.join(logger.log_dir, 'config.yaml')}")
    copy_python_files("src", os.path.join(logger.log_dir, "code", "src"))
    copy_python_files("python_scripts", os.path.join(logger.log_dir, "code", "python_scripts"))

    # setting up checkpoint saver
    checkpoint_callback = ModelCheckpoint(**conf.checkpointing.kwargs, dirpath=logger.log_dir)

    # Setting up progress bar
    progress_bar = TQDMProgressBar()

    # initialize trainer
    trainer = Trainer(logger=logger, **conf.trainer.kwargs, callbacks=[checkpoint_callback, progress_bar])

    trainer.fit(model, datamodule=datamodule, ckpt_path=conf.trainer.get("ckpt_path", None))


if __name__ == "__main__":
    main()
