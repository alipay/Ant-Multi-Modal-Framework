from torch import nn
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.models._utils import IntermediateLayerGetter


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone_type,
        backbone,
        return_layers,
        in_channels_list,
        out_channels,
        extra_blocks,
    ):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if backbone_type.lower() == "cctt":
            self.body = backbone
        elif backbone_type.lower().startswith("resnet"):
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            raise TypeError(
                f"backbone type must be cctt or resnet, but got {backbone_type}"
            )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
