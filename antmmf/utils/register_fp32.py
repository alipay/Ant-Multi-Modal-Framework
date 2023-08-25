import functools
import weakref
import torch
from antmmf.common.registry import registry
import warnings


def tensor2dtype(tensors, dtype):

    if isinstance(tensors, dict):
        keys = tensors.keys()
        values = [tensors[key] for key in keys]
        new_values = tensor2dtype(values, dtype)
        return {key: value for key, value in zip(keys, new_values)}
    if isinstance(tensors, (list, tuple)):
        rets = []
        for tensor in tensors:
            rets.append(tensor2dtype(tensor, dtype))
        return tuple(rets) if isinstance(tensors, tuple) else rets
    else:
        if torch.is_tensor(tensors) and torch.is_floating_point(tensors):
            return tensors.to(dtype)
        else:
            return tensors


def _new_forward(forward, ref_module, *args, **kwargs):

    with torch.cuda.amp.autocast(enabled=False):
        new_args = tensor2dtype(args, torch.float32)
        new_kwargs = tensor2dtype(kwargs, torch.float32)
        res = forward(ref_module(), *new_args, **new_kwargs)
    return res


def customed_forward(module):
    module.forward = functools.partial(
        _new_forward, type(module).forward, weakref.ref(module)
    )


def get_amp_escapes_name(amp_escapes):
    escapes_name = []
    for name in amp_escapes.split(","):
        name = name.strip()
        if name:
            escapes_name.append(name)
    return escapes_name


def set_escapes_class_fp32(model, amp_attributes):
    if hasattr(amp_attributes, "amp_escapes"):
        log_writer = registry.get("writer")
        escapes_name = get_amp_escapes_name(amp_attributes.amp_escapes)
        customed_num = 0
        for name, layer in model.named_modules():
            layer_name = type(layer).__name__
            if layer_name in escapes_name:
                customed_forward(layer)
                customed_num += 1
                log_writer.write(
                    f"layer run in fp32:{name} -> {type(layer).__name__}", log_all=True
                )
        if customed_num == 0:
            warnings.warn(
                "Can't find any class in model. config amp_escapes:{}".format(
                    amp_attributes.amp_escapes
                )
            )
