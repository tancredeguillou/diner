from pytorch_lightning import LightningModule
from src.util.import_helper import import_obj
from src.util.general import *
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os
from random import Random
import torch


class Regressor(LightningModule):
    def __init__(self, regressor_conf, lr=1e-4, img_log_interval=10000,
                 n_samples_score_eval=100):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dense_regressor = import_obj(regressor_conf.module)(**regressor_conf.kwargs)

        self.img_log_interval = img_log_interval
        self.n_samples_score_eval = n_samples_score_eval
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss_dict = self.dense_regressor.calc_losses(batch)
        loss_dict["step"] = float(self.global_step)
        batch_size = batch["target_rgb"].shape[0]
        self.log_dict(loss_dict, on_step=True, batch_size=batch_size)

        return loss_dict["total"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_dict = self.dense_regressor.calc_losses(batch)
        log_dict = prefix_dict_keys(loss_dict, "val_")
        log_dict["step"] = float(self.global_step)
        batch_size = batch["target_rgb"].shape[0]
        self.log_dict(log_dict, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss_dict["total"]

    @rank_zero_only
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dense_regressor.parameters(), lr=self.lr)
        return optimizer
