# Copyright (c) 2023 Ant Group and its affiliates.

"""
:class:`Sample` and :class:`SampleList` are data structures for arbitrary data returned from a
dataset. To work with antmmf, minimum requirement for datasets is to return
an object of :class:`Sample` class and for models to accept an object of type :class:`SampleList`
as an argument.

:class:`Sample` is used to represent an arbitrary sample from dataset, while :class:`SampleList`
is list of Sample combined in an efficient way to be used by the model.
In simple term, :class:`SampleList` is a batch of Sample but allow easy access of
attributes from :class:`Sample` while taking care of properly batching things.
"""

import torch
import collections
from typing import List, Any
from collections import OrderedDict


class Sample(OrderedDict):
    """Sample represent some arbitary data. All datasets in antmmf must
    return an object of type ``Sample``.

    Args:
        init_dict (dict): Dictionary to init :class:`Sample` class with.

    Usage::

        >>> sample = Sample({"text": torch.tensor(2)})
        >>> sample.text.zero_()
        # Custom attributes can be added to ``Sample`` after initialization
        >>> sample.context = torch.tensor(4)
    """

    def __init__(self, init_dict={}):
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())


class SampleList(OrderedDict):
    """:class:`SampleList` is used to collate a list of :class:`Sample` into a batch during batch
    preparation. It can be thought of as a merger of list of Dicts into a single Dict.

    If :class:`Sample` contains an attribute 'text' of size (2) and there are 10 samples in
    list, the returned :class:`SampleList` will have an attribute 'text' which is a tensor
    of size (10, 2).

    Args:
        samples (List): List of ``Sample`` from which the :class:`SampleList` will be created.

    Usage::

        >>> sample_list = SampleList([Sample({"text": torch.tensor(2)}), Sample({"text": torch.tensor(2)})])
        >>> sample_list.text
        torch.tensor([2, 2])
    """

    _TENSOR_FIELD_ = "_tensor_field"

    def __init__(self, samples: List = []):
        super().__init__(self)

        if len(samples) == 0:
            return

        if self._check_and_load_dict(samples):
            return
        # If passed sample list was in form of key, value pairs of tuples
        # return after loading these
        if self._check_and_load_tuple(samples):
            return

        fields = None
        informative_idx = -1
        for idx, sample in enumerate(samples):
            if sample is not None:
                fields = sample.keys()
                informative_idx = idx
                break

        if fields is None:
            return

        # effective size
        effect_size = len([s for s in samples if s is not None])

        for field in fields:
            if isinstance(samples[informative_idx][field], torch.Tensor):
                size = (effect_size, *samples[informative_idx][field].size())
                self[field] = samples[informative_idx][field].new_empty(size)
                if self._get_tensor_field() is None:
                    self._set_tensor_field(field)
            else:
                self[field] = [None for _ in range(effect_size)]

            idx = 0
            for sample in samples:
                # it should be a tensor but not a 0-d tensor
                if sample is None:
                    continue

                if (
                    isinstance(sample[field], torch.Tensor)
                    and len(sample[field].size()) != 0
                    and sample[field].size(0) != samples[informative_idx][field].size(0)
                ):
                    raise AssertionError(
                        f"Fields for all samples must be equally sized. {field} is of different sizes"
                    )

                self[field][idx] = sample[field]
                idx += 1

            if isinstance(samples[informative_idx][field], collections.abc.Mapping):
                self[field] = SampleList(self[field])

    def _check_and_load_tuple(self, samples):
        if isinstance(samples[0], (tuple, list)) and isinstance(samples[0][0], str):
            for kv_pair in samples:
                self.add_field(kv_pair[0], kv_pair[1])
            return True
        else:
            return False

    def _check_and_load_dict(self, samples):
        if isinstance(samples, collections.abc.Mapping):
            for key, value in samples.items():
                self.add_field(key, value)
            return True
        else:
            return False

    def _fix_sample_type(self, samples):
        if not isinstance(samples[0], Sample):
            samples = [Sample(sample) for sample in samples]
        return samples

    def __getattr__(self, key) -> Sample:
        if key not in self:
            raise AttributeError(
                "Key {} not found in the SampleList. Valid choices are {}".format(
                    key, self.fields()
                )
            )
        fields = self.keys()

        if key in fields:
            return self[key]

        sample = Sample()

        for field in fields:
            sample[field] = self[field][key]

        return sample

    def get_item_list(self, key: str):
        """Get :class:`SampleList` of only one particular attribute that is present
        in the :class:`SampleList`.

        Args:
            key (str): Attribute whose :class:`SampleList` will be made.

        Returns:
            SampleList: SampleList containing only the attribute value of the key
            which was passed.

        """
        sample = self[key]

        return SampleList([sample])

    def copy(self) -> "SampleList":
        """Get a copy of the current SampleList

        Returns:
            SampleList: Copy of current SampleList.

        """
        sample_list = SampleList()

        # TODO: this only copies keys but not attributes
        # for example, self.dataset_type is not copied
        # hence, need to have a way to copy everything, including attributes
        fields = self.fields()

        for field in fields:
            sample_list.add_field(field, self[field])

        return sample_list

    def fields(self) -> List[str]:
        """Get current attributes/fields registered under the SampleList.

        Returns:
            List[str]: list of attributes of the SampleList.

        """
        return list(self.keys())

    def get_fields(self, fields: List[str]) -> "SampleList":
        """Get a new ``SampleList`` generated from the current ``SampleList``
        but contains only the attributes passed in `fields` argument

        Args:
            fields (List[str]): Attributes whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute values of the fields
            which were passed.

        """
        current_fields = self.fields()

        return_list = SampleList()

        for field in fields:
            if field not in current_fields:
                raise AttributeError(
                    "{} not present in SampleList. Valid choices are {}".format(
                        field, current_fields
                    )
                )
            return_list.add_field(field, self[field])

        return return_list

    def get_field(self, field) -> Any:
        """Get value of a particular attribute

        Args:
            field (str): Attribute whose value is to be returned.
        """
        return self[field]

    def _get_tensor_field(self):
        return self.__dict__.get(SampleList._TENSOR_FIELD_, None)

    def _set_tensor_field(self, value):
        self.__dict__[SampleList._TENSOR_FIELD_] = value

    def get_batch_size(self) -> int:
        """Get batch size of the current ``SampleList``. There must be a tensor
        field present in the ``SampleList`` currently.

        Returns:
            int: Size of the batch in ``SampleList``.

        """
        tensor_field = self._get_tensor_field()
        assert tensor_field is not None, "There is no tensor yet in SampleList"

        return self[tensor_field].size(0)

    def add_field(self, field, data):
        """Add an attribute ``field`` with value ``data`` to the SampleList

        Args:
            field (str): Key under which the data will be added.
            data (object): Data to be added, can be a ``torch.Tensor``, ``list``
                         or ``Sample``
        """
        fields = self.fields()
        tensor_field = self._get_tensor_field()

        if len(fields) == 0:
            self[field] = data
        else:
            if (
                isinstance(data, torch.Tensor)
                and len(data.size()) != 0
                and tensor_field is not None
                and data.size(0) != self[tensor_field].size(0)
            ):
                raise AssertionError(
                    "A tensor field to be added must have same size as existing tensor "
                    "fields in SampleList. "
                    f"Passed size: {len(data)}, Required size: {len(self[fields[0]])}"
                )
            self[field] = data

        if isinstance(self[field], torch.Tensor) and tensor_field is None:
            self._set_tensor_field(field)

    def to(self, device, non_blocking=True) -> "SampleList":
        """Similar to ``.to`` function on a `torch.Tensor`. Moves all of the
        tensors present inside the ``SampleList`` to a particular device. If an
        attribute's value is not a tensor, it is ignored and kept as it is.

        Args:
            device (str|torch.device): Device on which the ``SampleList`` should
                                       moved.
            non_blocking (bool): Whether the move should be non_blocking. Default: True

        Returns:
            SampleList: a SampleList moved to the :external:py:class:`torch.device`.

        """
        fields = self.keys()
        sample_list = self.copy()
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError(
                    "device must be either 'str' or 'torch.device' type, {} found".format(
                        type(device)
                    )
                )
            device = torch.device(device)

        for field in fields:
            if hasattr(sample_list[field], "to"):
                sample_list[field] = sample_list[field].to(
                    device, non_blocking=non_blocking
                )

        return sample_list
