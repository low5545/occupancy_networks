import os
import torch.utils.data
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        classes,
        batch_size
    ):
        super().__init__()

        self.path = path
        self.classes = classes
        self.batch_size = batch_size
        
    def setup(self, stage):
        if stage in (None, "fit"):
            self.train_dataset = self._build_dataset("train")
            self.val_dataset = self._build_dataset("val")
        if stage in (None, "validate"):
            self.val_dataset = self._build_dataset("val")

    def _build_dataset(self, stage):
        data = None
        for class_id in self.classes:
            dataset_filepath = os.path.join(
                self.path, class_id, f"{stage}_latent_code.pt"
            )
            class_data = torch.load(dataset_filepath)
            if data is None:
                data = class_data
            else:
                data = torch.cat([ data, class_data ], dim=0)

        return torch.utils.data.TensorDataset(data)

    def train_dataloader(self):
        """
        NOTE:
            `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
            ...)` implicitly replaces the `sampler` of `dataset_dataloader`
            with `DistributedSampler(shuffle=True, drop_last=False, ...)`
        """
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        """
        NOTE:
            1. `pl.Trainer(replace_sampler_ddp=True, accelerator=ddp/ddp_spawn, 
               ...)` implicitly replaces the `sampler` of `dataset_dataloader`
               with `DistributedSampler(shuffle=True, drop_last=False, ...)`.
            2. If `len(self.val_dataset)` is not divisible by `num_replicas`,
               validation is not entirely accurate, irrespective of
               `drop_last`.
        """
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
        )
    