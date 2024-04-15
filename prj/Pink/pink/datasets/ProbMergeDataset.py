from typing import List, Optional, Literal

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset as TorchConcatDataset
import bisect
import random


class ProbMergeDataset(Dataset):
    def __init__(
            self,
            datasets = None,
            probabilities: Optional[List[float]] = None,
            seed: Optional[int] = 42,
            stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted",
    ):
        self.probabilities = probabilities
        self.seed = seed
        self.stopping_strategy = stopping_strategy

        self.concat_dataset = TorchConcatDataset(datasets)

        self.index_mapping = _interleave_dataset_index(
            lengths=[len(ds) for ds in datasets],
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, index):
        return self.concat_dataset[self.index_mapping[index]]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints: {self.__len__()}",
            f"Probabilities: {self.probabilities}",
            f"stopping_strategy: {self.stopping_strategy}",
            f"seed: {self.seed}",
        ]
        for i, ds in enumerate(self.concat_dataset.datasets):
            body.append(f"Subset {i + 1}/{len(self.concat_dataset.datasets)}")
            body += ds.__repr__().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


# stolen from huggingface/datasets
# https://github.com/huggingface/datasets/blob/074925b9b7c1dfd33b8675aa99c07cc26375665c/src/datasets/arrow_dataset.py#L5987
def _interleave_dataset_index(
        *,
        lengths: List[int],
        probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
):
    if probabilities is not None and 0 in probabilities:
        assert stopping_strategy == 'first_exhausted', "you will meet a Infinite loop"
    # Let's now build the indices to pass to .select()
    offsets = np.cumsum([0] + lengths[:-1])

    # if stopping_strategy is "first_exhausted", it is an undersampling situation whereas it is an oversampling situation if it is "all_exhausted"
    oversampling = stopping_strategy == "all_exhausted"

    if probabilities is None and not oversampling:
        # Undersampling situation with cycling between each sources
        # Example:: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples

        # Reasoning behind the following operation: keeping the min_length first indices of each dataset
        # while offsetting in order to correspond to the right indices of the concatenated dataset
        # and flattening to effectively interleave the datasets
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    elif probabilities is None:
        # Oversampling situation with cycling between each sources
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 5, 9, 0, 6, 10, 1, 3, 11]
        # Note that we have 5 examples per dataset with a rolling window since the longest dataset has 5 samples

        # Reasoning behind the following operation: for each dataset indices (i.e column) repeat the indices to have max_length indices per dataset
        # For example, if the max_length is 5 and the i-th dataset has 3 samples, the i-th column will be [0,1,2,0,1]
        indices = np.mod(np.arange(max(lengths)).reshape(-1, 1), np.array(lengths).reshape(1, -1))

        # We have to keep the indices to their respective dataset offsets and to flatten to effectively interleave the datasets
        indices = (indices + offsets).flatten().tolist()

    else:
        # boolean array indicating if at index i if the dataset_i has been fully exhausted
        is_exhausted = np.full(len(lengths), False)

        # if undersampling ("first_exhausted"), we stop as soon as one dataset is exhausted
        # if oversampling ("all_exhausted"), we stop as soons as every dataset is exhausted, i.e as soon as every samples of every dataset has been visited at least once
        bool_strategy_func = np.all if oversampling else np.any

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(lengths), size=1000, p=probabilities))

        current_index = [0] * len(lengths)
        indices = []
        for source_idx in iter_random_indices():
            # If no oversampling, we stop as soon as a dataset has ran out of examples (np.any)
            # Otherwise, we stop as soon as every dataset has ran out of examples (np.all)
            if bool_strategy_func(is_exhausted):
                # the stopping condition was reached, let's stop
                break

            # let's add the example at the current index of the `source_idx`-th dataset
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] += 1

            # we've ran out of examples for the current dataset, let's update our boolean array and bring the current_index back to 0
            if current_index[source_idx] >= lengths[source_idx]:
                is_exhausted[source_idx] = True
                current_index[source_idx] = 0
    return indices


class MemoryEfficientConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, repeats=None) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        # for d in self.datasets:
        #     assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        
        self.repeats = repeats
        if repeats is not None:
            self.datasets_2_real_index = {}
            self.datasets_length = []
            num_datasets = 0
            for i, r in enumerate(repeats):
                for _ in range(r):
                    self.datasets_2_real_index[num_datasets] = i
                    num_datasets += 1
                    self.datasets_length.append([0 for _ in range(len(self.datasets[i]))])
            self.cumulative_sizes = self.cumsum(self.datasets_length)
        else:
            self.cumulative_sizes = self.cumsum(self.datasets)
        print(self.cumulative_sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        if self.repeats is not None:
            dataset_idx = self.datasets_2_real_index[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class ProbConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]

    def __init__(self, datasets, real_epoch=6, probs=None) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        # for d in self.datasets:
        #     assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.shortest_length = 1e16
        self.shortest_index = -100
        for index, d in enumerate(self.datasets):
            if len(d) < self.shortest_length:
                self.shortest_length = len(d)
                self.shortest_index = index

        self.real_epoch = real_epoch
        self.merge_length = self.shortest_length * self.real_epoch
        assert probs is not None and len(probs) == len(self.datasets) and sum(probs) == 1
        self.probs = probs
        basic_length = self.merge_length // (probs[self.shortest_index])

        sample_length = []
        for prob in self.probs:
            sample_length.append(int(basic_length * prob))
        self.sample_length = sample_length

        self.sample_list = []

        for index, d in enumerate(self.datasets):
            dataset_list = []
            sample_length = self.sample_length[index]
            current_length = 0
            while current_length < sample_length:
                if (sample_length - current_length) > len(d):
                    dataset_list += [(index, i) for i in range(len(d))]
                    current_length += len(d)
                else:
                    random_sample_list = random.sample(range(len(d)), sample_length - current_length)
                    dataset_list += [(index, i) for i in random_sample_list]
                    current_length += len(random_sample_list)
            self.sample_list += dataset_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.sample_list[idx]

        return self.datasets[dataset_idx][sample_idx]
