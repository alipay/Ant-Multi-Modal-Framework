# -- coding: utf-8 --
# Copyright (c) 2023 Ant Group and its affiliates.

from torch import nn
from antmmf.common.registry import registry


@registry.register_loss("discount_infonce")
class DInfoNCELoss(nn.Module):
    """
    Noise Constrastive Estimation Loss

    The loss is computed in SampledSoftmaxLoss, and is
    used inside a model as a scorer.
    The scorer is used as a last layer to output "sampled" softmax.
    The sampling is done on the output label indices.
    Because of this, computational cost for normalization is reduced.
    In evaluation mode, all of label indices are used to have exact softmax.

    An example of using NCE loss is in MMFusion model.
    Notice that it has a scorer that computes NCE loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output, *args, **kwargs):
        temperature = args[0]
        assert "class_emb" in model_output
        class_emb = model_output["class_emb"]
        assert "output" in model_output
        output = model_output["output"]
        assert "targets" in sample_list
        targets = sample_list["targets"]
        if targets.size()[-1] < class_emb.size()[0]:
            targets_extra = torch.sum(
                hard_labels, axis=-1, keepdims=True
            )  # [batch_size,1]
            targets_extra = torch.where(
                hard_labels_extra > 0,
                x=torch.zeros_like(targets_extra),
                y=torch.ones_like(targets_extra),
            )  # [batch_size,1]
            targets = torch.cat(
                [targets, targets_extra], axis=-1
            ).float()  # [batch_size, num_classes]
        num_classes = targets.size()[-1]

        mask_count = torch.sum(targets, 0)
        output = torch.tile(
            torch.unsqueeze(output, 1), [1, num_classes, 1]
        )  # [batch_size, num_classes, hidden_size]
        # [batch_size, num_classes, hidden_size]
        mask = torch.tile(torch.unsqueeze(targets, 2), [1, 1, output.shape[-1]])
        output = output * mask  # mask select [batch_size, num_classes, hidden_size]
        output = output.transpose(0, 1)  # [num_classes, batch_size, hidden_size]
        # [num_classes, hidden_size]
        output = torch.sum(output, 1) / (
            torch.count_nonzero(output, axis=1).float() + 1e-9
        )
        # [num_classes, num_classes, hidden_size]
        output = torch.tile(torch.unsqueeze(output, 1), [1, num_classes, 1])

        # [num_classes, num_classes, hidden_size]
        class_emb = torch.tile(torch.unsqueeze(class_emb, 0), [num_classes, 1, 1])
        all_cosine = get_cos_distance(output, class_emb)  # [num_classes, num_classes]
        class_sim = get_cos_distance(
            class_emb.transpose(0, 1), class_emb
        )  # [num_classes, num_classes]
        _, index = torch.top_k(
            -1 * class_sim, class_sim.shape[-1]
        )  # [num_classes, num_classes]
        rank = torch.argsort(index)  # [num_classes, num_classes]
        discount = torch.log((rank + 2).float())
        all_cosine = all_cosine / (temperature * discount)
        all_cosine_norm = torch.softmax(
            all_cosine, axis=1
        )  # [num_classes, num_classes]
        # [num_classes, ]
        all_cosine_norm_positive = torch.gather(
            all_cosine_norm, torch.where(torch.eye(num_classes))
        )
        # [select_num, ]
        infonce = -torch.log(
            torch.gather(all_cosine_norm_positive, torch.where(mask_count))
        )
        infonce_mean = torch.reduce_mean(infonce)  # [1,]

        return loss
