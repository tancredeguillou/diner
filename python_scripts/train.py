"""
Main script for training diner
"""

from pytorch3d.ops.knn import knn_points
from omegaconf import OmegaConf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.data.pl_datamodule import PlDataModule
from pytorch_lightning import Trainer
from src.models.diner import DINER
from src.models.novel.novel import NOVEL
from src.models.keypointnerf import KeypointNeRFLightningModule
from src.models.ournerf import OurNeRFLightningModule
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
from src.util.general import copy_python_files
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

def main():
    config_path = sys.argv[1]
    model_name = sys.argv[2]
    data_type = None
    if len(sys.argv) == 4:
        data_type = sys.argv[3]
    if model_name not in ['DINER', 'KeypointNeRF', 'NOVEL']:
        raise ValueError(f'Model Name should be DINER or KeypointNeRF but got: {model_name}')
    
    conf = OmegaConf.load(config_path)
    conf_logger = conf.logger
    os.makedirs(conf_logger.kwargs.save_dir, exist_ok=True)
    datamodule = PlDataModule(conf.data.train, conf.data.val, model_name, data_type)
    datamodule.setup()

    # initialize model
    if model_name == 'DINER':
        model = DINER(nerf_conf=conf.nerf, renderer_conf=conf.renderer, znear=datamodule.train_set.znear,
                              zfar=datamodule.train_set.zfar, **conf.optimizer_diner.kwargs)
    elif model_name == 'KeypointNeRF':
        model = KeypointNeRFLightningModule(conf.keypoint_nerf, znear=datamodule.train_set.znear,
                              zfar=datamodule.train_set.zfar, **conf.optimizer_keypointnerf.kwargs)
    elif model_name == 'NOVEL':
        model = NOVEL(nerf_conf=conf.nerf, renderer_conf=conf.renderer, regressor_conf=conf.regressor, znear=datamodule.train_set.znear,
                              zfar=datamodule.train_set.zfar, **conf.optimizer_diner.kwargs)
    else:
        raise ValueError(f'Model Name should be DINER or KeypointNeRF but got: {model_name}')

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
