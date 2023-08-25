from itertools import chain
from typing import List
from torch.utils.data.dataloader import DataLoader


class MultiDataLoader:
    def __init__(
        self,
        loader_list: List[DataLoader],
        task_loader,
        total_device_num: int = 1,
    ):
        self.loader_list = loader_list
        self.task_loader = task_loader
        self.total_device_num = total_device_num
        self.total_length = None
        self._loader_iter = None

    def __len__(self) -> int:
        if not self.total_length:
            self.total_length = 0
            for dataloader in self.loader_list:
                self.total_length += len(dataloader)
        return self.total_length // self.total_device_num

    def __iter__(self):
        self._loader_iter = iter(chain(*self.loader_list))
        return self

    def __next__(self):
        batch = next(self._loader_iter)
        batch = self.task_loader.prepare_batch(batch)
        return batch
