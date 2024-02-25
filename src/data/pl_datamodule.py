import pytorch_lightning as pl
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from src.util.import_helper import import_obj


class PlDataModule(pl.LightningDataModule):
    def __init__(self, train_config, val_config, model, data_type=None):
        """
        Pytorch Lightning Data Module Wrapper for datasets
        :param train_config:
        :param val_config:
        """
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.model = model
        self.data_type = data_type

        self.train_set = None
        self.val_set = None

    def setup(self, stage: Optional[str] = None):
        self.stage = stage

        # obtaining scan lists
        if stage == "train" or stage is None:
            dset_class = import_obj(self.train_config.dataset.module)
            self.train_set = dset_class(self.model, **self.train_config.dataset.kwargs, stage="train", data_type=self.data_type)

        if stage == "val" or stage is None:
            dset_class = import_obj(self.val_config.dataset.module)
            self.val_set = dset_class(self.model, **self.val_config.dataset.kwargs, stage="val", data_type=self.data_type)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self.train_config.dataloader.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self.val_config.dataloader.kwargs)
