# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import copy
import inspect
import warnings

import torch
from torch import Tensor
from typing import Dict, Any, Callable, Sized


class SizedDataStructure:
    """SizedDataStructure is used for some data structure which needs to implement the function of index select,
    setattr, getattr, iterable, and etc.

    Examples::

        boxes = SizedDataStructure()
        boxes.tensor = torch.rand(3, 4)
        boxes.score = torch.rand(3)             # set attribute for boxes
        assert len(boxes) == 3                  # can be called by len
        list_boxes = [box for box in boxes]     # can be iterable
        box = boxes[2]                          # use index to select the values

    """

    def __init__(self, tensor: Tensor, **kwargs):
        """
        Args:
            tensor (Tensor): the main data of the structure, like boxes or images
        """
        super(SizedDataStructure, self).__init__()
        self.tensor = tensor
        self.kwargs = kwargs
        self.__fields__: Dict[str, Any] = {}

    def clone(self):
        """
        Copy tensor and all other fields in the way of deepcopy.

        Returns:
            an instance of SizedDataStructure.
        """
        instance = type(self)(self.tensor.clone(), **copy.deepcopy(self.kwargs))
        for key, value in self.__fields__.items():
            setattr(instance, key, copy.deepcopy(value))

        return instance

    def to(self, **kwargs):
        """
        Move data to cuda or cpu, or convert data type.

        We use inspect to determine the variables for the function of to of tensor and all other fields,
        so you need to send all variables without default value of these functions.

        Example::

            data = SizedDataStructure(torch.rand(3, 4))
            data.journey = Journey()        # a mock class whose function of to has a variable named "position"
            data = data.to(device="cuda:0", position="ChengDu")
            print(data.tensor.device)       # cuda:0
            print(data.journey.position)    # ChengDu
        """

        def filter_parameters(func):
            func_parameters = inspect.signature(func).parameters
            return {k: v for k, v in kwargs.items() if k in func_parameters}

        self.tensor.to(**filter_parameters(self.tensor.to))
        for key, value in self.__fields__.items():
            if hasattr(value, "to") and isinstance(value.to, Callable):
                _kwargs = filter_parameters(value.to)
                self.__fields__[key] = value.to(**_kwargs)
        return self

    @property
    def device(self) -> torch.device:
        """
        return the device of the tensor.
        """
        return self.tensor.device

    def has_field(self, field: str) -> bool:
        """
        Args:
            field: field string

        Returns:
            bool: whether the field is existed or not
        """
        return field in self.__fields__

    def __len__(self):
        return len(self.tensor)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
           returns an `SizedDataStructure` where tensor and all other fields are selected by `item`.
        """
        if not isinstance(item, (int, slice, torch.LongTensor, torch.BoolTensor)):
            raise NotImplementedError(
                f"item must be int, slice, LongTensor and BoolTensor, but got {item} of type {type(item)}"
            )

        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError(f"index out of range, got {item} vs. {len(self)}")
            else:
                item = slice(item, None, len(self))

        if isinstance(item, torch.LongTensor):
            indices = item.tolist()
        elif isinstance(item, torch.BoolTensor):
            indices = torch.nonzero(item, as_tuple=True)[0]
        else:
            # item is slice
            indices = []

        fields = {}
        for k, v in self.__fields__.items():
            if isinstance(v, Tensor) or isinstance(item, slice):
                v = v[item]
            else:
                v = type(v)([v[idx] for idx in indices])
            fields[k] = v

        kwargs = {k: v for k, v in self.kwargs.items()}
        arguments = inspect.signature(self.__init__).parameters
        for k in list(fields.keys()):
            if k in arguments:
                kwargs[k] = fields.pop(k)

        ret = type(self)(self.tensor[item].clone(), **kwargs)
        for k, v in fields.items():
            setattr(ret, k, copy.deepcopy(v))

        # copy the other attribute to the ret
        for k, v in self.__dict__.items():
            if k not in ret.__dict__:
                setattr(ret, k, v)

        return ret

    def __setattr__(self, key: str, value: Any):
        if (
            key in self.__dict__
            or isinstance(value, (str, dict))
            or not isinstance(value, Sized)
            or key == "tensor"
        ):
            return super(SizedDataStructure, self).__setattr__(key, value)

        if len(value) != len(self.tensor):
            raise AttributeError(
                f"The length of value with key of {key} must be same as the number of self, "
                f"got {len(value)} vs. {len(self.tensor)}"
            )

        self.__fields__[key] = value

    def __getattr__(self, key, default_value=None):
        if key in self.kwargs:
            return self.kwargs[key]

        if key in self.__fields__:
            return self.__fields__[key]

        if default_value is None:
            raise AttributeError(f"{key} is not exist.")

        if not isinstance(default_value, Sized):
            warnings.warn(
                f"You are input a non-exist key: {key}, and the default_value: `{default_value}` "
                f"is not a sized instance."
            )

        if isinstance(default_value, Sized) and len(default_value) != len(self):
            warnings.warn(
                f"You are input a non-exist key: {key}, and the length of default_value is not equal to "
                f"len({type(self)}), expect {len(self)}, but got {len(default_value)}."
            )

        return default_value

    def __str__(self) -> str:
        s = self.__class__.__name__ + "(\n"
        s += f"  tensor={self.tensor.shape}, \n"
        s += f"  kwargs={self.kwargs} \n"
        s += "  fields=[\n{}\n  ]\n)".format(
            ", \n".join(
                (
                    f"    {k}: {v.shape if isinstance(v, Tensor) else v}"
                    for k, v in self.__fields__.items()
                )
            )
        )
        return s

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += f"  tensor={self.tensor}, \n"
        s += f"  kwargs={self.kwargs} \n"
        s += "  fields=[\n{}\n  ]\n)".format(
            ", \n".join((f"    {k}: {v}" for k, v in self.__fields__.items()))
        )
        return s
