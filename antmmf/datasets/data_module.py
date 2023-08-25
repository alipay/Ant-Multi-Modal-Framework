import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(
        self, train_dataloader=None, val_dataloader=None, test_dataloader=None
    ):
        super().__init__()
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
