import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from vilt.datasets import WitDataset
from torch.utils.data.distributed import DistributedSampler


class WitDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.max_text_len = _config['max_text_len']
        self.setup_flag = False
        self.train_transform_keys = (
            ["wit_default"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dist = dist

        self.roberta = torch.hub.load('pytorch/fairseq:2f7e3f3323', 'roberta.base')

    @property
    def dataset_cls(self):
        return WitDataset

    @property
    def dataset_cls_no_false(self):
        return WitDataset

    @property
    def dataset_name(self):
        return "wit"

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.roberta
            self.val_dataset.tokenizer = self.roberta
            self.test_dataset.tokenizer = self.roberta

            self.setup_flag = True

            if self.dist:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.train_sampler = None
                self.val_sampler = None
                self.test_sampler = None

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.train_transform_keys,
            split="train",
            max_text_len=self.max_text_len
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.train_transform_keys,
            split="val",
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            transform_keys=self.train_transform_keys,
            split="test",
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader

