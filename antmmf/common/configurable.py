# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

import inspect
import functools
from antmmf.common.configuration import Configuration


def configurable(init_func=None, *, from_config=None):
    """
    Many of following codes are referenced from
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/config.py, and it is used to replace
    the :class:`~.configuration.AntMMFConfig` to filter the unused configurations, furthermore, it is
    flexible to construct instances with different configurations.

    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`~.configuration.Configuration` object using a :func:`from_config` function that translates
    :class:`~.configuration.Configuration` to arguments. Without the :func:`from_config`, configurable will inject
    :func:`default_from_config` as the :func:`from_config` of your class.

    .. warning::

        If your child and parent classes are both decorated by configurable, and your parent class has its own
        custom `from_config`, you *should not* use the way of *super().__init__(config)* in your __init__ of
        the child class, but use super().__init__(a=xxx, b=xxx) instead. Otherwise it will cause some unexpect errors.

        For example:

        .. code-block:: python

            class A:
                @configurable
                def __init__(self, a, b=2, c=3):
                    pass

            class B(A):
                @configurable
                def __init__(self, a, b=2, d=5):
                    # risk code:
                    #  config = Configuration(locals())     # use local variables as configuration
                    #  super().__init__(config)             # will raise TypeError
                    super().__init__(a=a, b=b)              # right code

    Examples:

    .. code-block:: python

        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, config):   # 'config' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": config.A, "b": config.B}
        a1 = A(a=1, b=2)            # regular construction
        a2 = A(config)              # construct with a config
        a3 = A(config, b=3, c=4)    # construct with extra overwrite

        # Usage 2: Decorator on __init__ without from_config:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            # use :func:`default_from_config` as `from_config`
        a1 = A(a=1, b=2)            # regular construction
        a2 = A(config)              # construct with a config
        a3 = A(config, b=3, c=4)    # construct with extra overwrite

        # Usage 3: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda config: {"a: config.A, "b": config.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)           # regular call
        a2 = a_func(config)             # call with a config
        a3 = a_func(config, b=3, c=4)   # call with extra overwrite

    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class should have a ``from_config`` classmethod which takes `config` as
            the first argument, otherwise, we will filter out the unused variable(s)
            from the config by `inspect`.
        from_config (callable): the from_config function in usage 2. It must take `config`
            as its first argument.
    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError:
                type(self).from_config = classmethod(default_from_config)
                from_config_func = type(self).from_config
            if not inspect.ismethod(from_config_func):
                raise TypeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                )

            if _called_with_config(*args, **kwargs):
                explicit_args = _get_args_from_config(
                    from_config_func, init_func, *args, **kwargs
                )
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_config(*args, **kwargs):
                    explicit_args = _get_args_from_config(
                        from_config, orig_func, *args, **kwargs
                    )
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper


def default_from_config(cls, config: Configuration, **kwargs):
    """
    Inspect the arguments of :func:`cls.__init__` and filter out the unused configurations.

    Returns:
         dict: arguments to be used for :func:`cls.__init__`
    """
    parameters = inspect.signature(cls.__init__).parameters
    if "kwargs" in parameters:
        return {
            k: config.get(k) for k in config.keys() if k not in ["self", "__class__"]
        }

    ret = {
        k: config.get(k)
        for k in parameters
        if k in config and k not in ["self", "__class__"]
    }
    ret.update(kwargs)
    return ret


def _get_args_from_config(from_config_func, init_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for :func:`cls.__init__`
    """
    parameters = inspect.signature(from_config_func).parameters
    if list(parameters.keys())[0] != "config":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'config' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in parameters.values()
    )
    if (
        support_var_arg
    ):  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)

    parameters = inspect.signature(init_func).parameters
    if "kwargs" not in parameters:
        ret = {k: v for k, v in ret.items() if k in parameters}
    return ret


def _called_with_config(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain :class:`Configuration` and should be considered
        forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (Configuration, DictConfig)):
        return True
    if isinstance(kwargs.pop("config", None), (Configuration, DictConfig)):
        return True
    # `from_config`'s first argument is forced to be "config".
    # So the above check covers all cases.
    return False
