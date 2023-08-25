"""
Using faster op,speed up training
"""
import torch
from torch.nn import LayerNorm, Linear, GELU
from torch.nn import MultiheadAttention, Sequential
import warnings

try:
    from transformers.models.bert.modeling_bert import BertAttention

    replace_transformer_bert = True

except ImportError:
    replace_transformer_bert = False


class DefaultStrategy:
    replace_mha = True
    replace_layernorm = True
    replace_linear_gelu = False


def replace_layer_norm(module: torch.nn.Module, cur_name: str):

    for name, child in module.named_children():
        child_name = cur_name + "." + name
        if isinstance(child, LayerNorm):
            new_module = FastLayerNorm(child.normalized_shape, eps=child.eps)
            new_module.load_state_dict(child.state_dict())
            setattr(module, name, new_module)
        else:
            replace_layer_norm(child, child_name)


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())


def _fused_dense_gelu_dense(input, weight1, bias1, weight2, bias2):
    batch, seq_length, hidden_size = input.size()
    input = input.view(batch * seq_length, hidden_size)
    args = _cast_if_autocast_enabled(input, weight1, bias1, weight2, bias2)
    from apex.fused_dense import FusedDenseGeluDenseFunc  # with cast

    with torch.cuda.amp.autocast(enabled=False):
        out = FusedDenseGeluDenseFunc.apply(*args)
        out = out.view(batch, seq_length, -1)
        return out


def linear_gelu_forward(input_, weight1, bias1, weight2, bias2):
    return _fused_dense_gelu_dense(input_, weight1, bias1, weight2, bias2)


def replace_linear_gelu(module, cur_name: str):
    """
    (layers): Sequential(
              (0): Sequential(
                (0): Linear(in_features=1536, out_features=6144, bias=True)
                (1): GELU()
                (2): Dropout(p=0.0, inplace=False)
              )
              (1): Linear(in_features=6144, out_features=1536, bias=True)
              (2): Dropout(p=0.0, inplace=False)
            )
    """
    for name, child in module.named_children():
        child_name = cur_name + "." + name
        if isinstance(child, Sequential):
            if (
                len(child) >= 2
                and isinstance(child[0], Sequential)
                and isinstance(child[1], Linear)
                and len(child[0]) >= 2
                and isinstance(child[0][0], Linear)
                and isinstance(child[0][1], GELU)
            ):  # Sequential+Linear
                linear0 = child[0][0]
                linear1 = child[1]
                if getattr(child, "replace_linear_gelu", False):
                    continue
                child.forward = lambda x: linear_gelu_forward(
                    x, linear0.weight, linear0.bias, linear1.weight, linear1.bias
                )
                child.replace_linear_gelu = True
                print("REPLACE linear+gelu:%s" % child_name)
            # setattr(module, name, new_module)
        else:
            replace_linear_gelu(child, child_name)
