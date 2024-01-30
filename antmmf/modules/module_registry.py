# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import inspect
from typing import ClassVar
from torch import nn


class ModuleRegistry(nn.Module):
    """A registry for module. You can extent this registry to any type of module you need, and the registry will
    help you to find the registered module according to argument of `module_type`, and construct them by `args`
    and `kwargs`. Finally, this registry will create an module instance which contain an attribute `_module`,
    and it is not necessary to access this module to operate them. Instead, you can use this instance just same as
    :external:py:class:`torch.nn.Module`, and the details are shown as follows::

        @ModuleRegistry.register()
        class MLP(torch.nn.Module):
            def __init__(self, input_dim, output_dim, hidden_dim)
                ...

        layer = ModuleRegistry("MLP", 64, 128, hidden_dim=256)
        x: torch.Tensor = ...
        output = layer(x)

    Args:
        module_type (str): the type of module, like `MLP` which shown in above.
        args: your ordered arguments of `module_type`.
        kwargs: your dictionary like arguments of `module_type`.
    """

    __register_module__ = {}

    @classmethod
    def register(cls, module: ClassVar = None):
        """Register a module which must be a class.

        Example1::

            @ModuleRegistry.register
            class MLP(torch.nn.Module):
                def __init__(self, in_dim, out_dim, hidden_dim):
                    ...

        Example2::

            @ModuleRegistry.register()
            class MLP(torch.nn.Module):
                def __init__(self, in_dim, out_dim, hidden_dim):
                    ...

        Example3::

            class MLP(torch.nn.Module):
                def __init__(self, in_dim, out_dim, hidden_dim):
                    ...

            ModuleRegistry.register(MLP)
        """
        # make ClassifierLayer.register() is equal to ClassifierLayer.register
        if module is None:
            return cls.register

        if not inspect.isclass(module):
            raise ValueError(
                f"Only class can be registered, but got {module} with type of `{type(module)}`."
            )
        cls.__register_module__[module.__name__] = module
        return module

    @classmethod
    def get(cls, module_type: str) -> ClassVar:
        if module_type not in cls.__register_module__:
            raise ValueError(f"{module_type} is not registered in {cls.__name__}.")

        return cls.__register_module__[module_type]

    def __init__(self, module_type: str, *args, **kwargs):
        super(ModuleRegistry, self).__init__()
        self.module = type(self).get(module_type)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
