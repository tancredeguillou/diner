import torch
import tqdm
from pytorch_lightning import LightningModule
from src.util.import_helper import import_obj
from src.util.general import *
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os
from random import Random
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.utilities import move_data_to_device
import numpy as np


class Regressor(LightningModule):
    def __init__(self, regressor_conf, lr=1e-4, img_log_interval=10000,
                 n_samples_score_eval=300):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dense_regressor = import_obj(regressor_conf.module)(**regressor_conf.kwargs)

        self.img_log_interval = img_log_interval
        self.n_samples_score_eval = n_samples_score_eval
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss_dict = self.dense_regressor.calc_losses(batch)
        loss_dict["step"] = float(self.global_step)
        batch_size = batch["image"].shape[0]
        self.log_dict(loss_dict, on_step=True, batch_size=batch_size)

        return loss_dict["total"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_dict = self.dense_regressor.calc_losses(batch)
        log_dict = prefix_dict_keys(loss_dict, "val_")
        log_dict["step"] = float(self.global_step)
        batch_size = batch["image"].shape[0]
        self.log_dict(log_dict, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss_dict["total"]
    
    @torch.no_grad()
    def create_prediction_folder(self, outdir, return_last_pred=False, dataloader=None):
        os.makedirs(outdir, exist_ok=True)

        # creating dataloader
        if dataloader is None:
            dataset = self.trainer.datamodule.val_set
            datalen = len(dataset)
            example_loader = self.trainer.datamodule.val_dataloader()
            batch_size = 1
            num_workers = 0
            sample_idcs = list(range(datalen))
            if self.n_samples_score_eval > 0 and self.n_samples_score_eval < datalen:
                sample_idcs = Random(0).sample(sample_idcs, self.n_samples_score_eval)
            sample_idcs = torch.tensor(sample_idcs).int()
            batch_sampler = BatchSampler(sample_idcs, batch_size=batch_size, drop_last=False)
            dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)

        # predicting and writing images & data
        scores = dict()
        for batch in tqdm.tqdm(dataloader):
            batch = move_data_to_device(batch, self.device)
            loss_dict = self.dense_regressor.calc_losses(batch)
            scores[batch["sample_name"][0]] = loss_dict["total"].item()
        
        average = dict()
        scores_list = list(scores.values())
        average['L1'] = np.mean(scores_list).astype(np.float64)

        # writing average metrics
        summary_fp = os.path.join(outdir, "average_scores.json")
        with open(summary_fp, "w") as f:
            json.dump(average, f, indent="\t")

        # writing detailed metrics report
        detail_fp = os.path.join(outdir, "detailed_report.json")
        with open(detail_fp, "w") as f:
            json.dump(scores, f, indent="\t")
            
        return average

    @rank_zero_only
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        if self.global_step > 0:
            eval_dir = os.path.join(self.trainer.log_dir, f"eval_{self.trainer.global_step:06d}")
            os.makedirs(eval_dir, exist_ok=True)

            # saving checkpoint
            self.trainer.save_checkpoint(os.path.join(eval_dir, f"{self.trainer.global_step:06d}.ckpt"))

            # creating and evaluating visualizations
            # visdir = os.path.join(eval_dir, "visualizations")
            score_dict = self.create_prediction_folder(outdir=eval_dir)
            # score_dict = eval_suite.evaluate_folder(visdir, eval_dir)
            log_dict = prefix_dict_keys(score_dict, "valscores_")
            log_dict["step"] = float(self.global_step)
            self.log_dict(log_dict, rank_zero_only=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dense_regressor.parameters(), lr=self.lr)
        return optimizer
